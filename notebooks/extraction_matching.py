import json
import random
import sys
import time
import threading
from ast import literal_eval
from pathlib import Path
from queue import Queue
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

__all__ = [
    'two_step_extraction',
    'compile_extracted_results',  # This is now deprecated, use run_extraction
    'match_results',
    'run_extraction',
    'process_file'
]

# =======================
# Prompts (unchanged)
# =======================

# NOTE: The prompts below use non-standard `{{` and `}}` for dictionary examples.
# The `_parse_json_result` function uses `ast.literal_eval` as a fallback
# to handle this syntax, as it is not valid JSON.

prompt_step1 = """
From the following given title and abstract, extract all studied exposures and outcomes separately. Ignore any other information.

Extract the following information:

1. "exposures": a list of all studied exposures. Be specific, for example, for salt intake estimated by 24 hour urinary sodium, use 24 hour urinary sodium.Return in full names. And use standard vocabularies.
2. "outcomes": a list of all studied outcomes. Return in full names. And use standard vocabularies.

Return the extracted information in a JSON dictionary.

Example input:
""
Title: "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion"

Abstract: "Context: Extrapolations from observational studies and short-term intervention trials suggest that population-wide moderation of salt intake might reduce cardiovascular events. Objective: To assess whether 24-hour urinary sodium excretion predicts blood pressure (BP) and health outcomes. Design, setting, and participants: Prospective population study, involving 3681 participants without cardiovascular disease (CVD) who are members of families that were randomly enrolled in the Flemish Study on Genes, Environment, and Health Outcomes (1985-2004) or in the European Project on Genes in Hypertension (1999-2001). Of 3681 participants without CVD, 2096 were normotensive at baseline and 1499 had BP and sodium excretion measured at baseline and last follow-up (2005-2008). Main outcome measures: Incidence of mortality and morbidity and association between changes in BP and sodium excretion. Multivariable-adjusted hazard ratios (HRs) express the risk in tertiles of sodium excretion relative to average risk in the whole study population. Results: Among 3681 participants followed up for a median 7.9 years, CVD deaths decreased across increasing tertiles of 24-hour sodium excretion, from 50 deaths in the low (mean, 107 mmol), 24 in the medium (mean, 168 mmol), and 10 in the high excretion group (mean, 260 mmol; P < .001), resulting in respective death rates of 4.1% (95% confidence interval [CI], 3.5%-4.7%), 1.9% (95% CI, 1.5%-2.3%), and 0.8% (95% CI, 0.5%-1.1%). In multivariable-adjusted analyses, this inverse association retained significance (P = .02): the HR in the low tertile was 1.56 (95% CI, 1.02-2.36; P = .04). Baseline sodium excretion predicted neither total mortality (P = .10) nor fatal combined with nonfatal CVD events (P = .55). Among 2096 participants followed up for 6.5 years, the risk of hypertension did not increase across increasing tertiles (P = .93). Incident hypertension was 187 (27.0%; HR, 1.00; 95% CI, 0.87-1.16) in the low, 190 (26.6%; HR, 1.02; 95% CI, 0.89-1.16) in the medium, and 175 (25.4%; HR, 0.98; 95% CI, 0.86-1.12) in the high sodium excretion group. In 1499 participants followed up for 6.1 years, systolic blood pressure increased by 0.37 mm Hg per year (P < .001), whereas sodium excretion did not change (-0.45 mmol per year, P = .15). However, in multivariable-adjusted analyses, a 100-mmol increase in sodium excretion was associated with 1.71 mm Hg increase in systolic blood pressure (P.<001) but no change in diastolic BP. Conclusions: In this population-based cohort, systolic blood pressure, but not diastolic pressure, changes over time aligned with change in sodium excretion, but this association did not translate into a higher risk of hypertension or CVD complications. Lower sodium excretion was associated with higher CVD mortality.
""

Example output:
{
    "exposures": [“24-hour urinary sodium excretion”],
    "outcomes": [“systolic blood pressure changes”, “diastolic blood pressure changes”, “incidence of hypertension”, “CVD mortality”, “total mortality”, “fatal and nonfatal cardiovascular disease events”]
}

Only output in JSON format, do not include any explanations or additional text in your response.
"""

prompt_step2_template = """
From the following given title and abstract, extract the primary relationships between the given exposures and outcomes.

Use the provided entities as references.

Extract the following information:

1. "exposure": the studied exposure
2. "exposure_direction": based on context, classify the direction of the exposure to the level of salt or urinary sodium, whether it's increased or decreased (only increased or decreased) (exposures without explict of decreased expression is an increased). by 'increased' it means higher level of exposure/intervention and 'decreased' it means lower level of exposure/intervention. for example, salt restriction/substitute is a 'decreased'. dietary intervention/education is a decreased (because it changes diet to lower salt intake). but you should also be cautious on for example less dietary intervention/education is an increased (opposite of decreased). refer to instruction in the end of prompt.
3. "outcome": the studied outcome
4. "significance": only return 'positive' if there exists STATISTICALLY significance or 'negative' if no STATISTICAL significance.
5. "direction": return whether th level of outcome increases, decreases, or no_change with the exposure in this relationship (no_change include relationships such as indifferent or unchanged)(only return one of the 3 categories: increase, decrease, no_change)
6. "population_main_condition": disease of studied population
7. "comparator": compared group in relationship
8. "study_design": (only 1 of 7 strict abbreiviation categories: Mendelian randomization ("MR"), Randomized controlled trial("RCT"), Observational study("OS"),Meta analysis ("META"), Review ("REVIEW"), Systematic Review ("SR"),  Others);
9. "included studies": (only return number + RCT/MR/OS)(only for META/REVIEW/SR, strict null for others(RCT,MR,OS));
10. "number of participants": (numeric number only) number of enrolled subjects

Return the extracted information in a list of flat JSON dictionaries of each result.

*Exposure Direction Classification
For each exposure mentioning salt or sodium intake:
    1. If the intervention or action explicitly decreases sodium/salt (e.g., “salt restriction,” “salt substitute,” “low-sodium diet,” “reduce salt”), set "exposure_direction" to "decreased".
    2. If it explicitly increases sodium/salt, or uses standard/high salt (e.g., “high-salt diet,” “less salt education,” “usual diet with no reduction”), set "exposure_direction" to "increased".

    Examples of Phrases → exposure_direction:
    •  “Participants in the intervention arm received a salt-substitution product” → decreased
    •  “Control arm participants received the usual advice without further diet restrictions” → increased
    •  “We advised a low-sodium diet” → decreased
    •  “We gave participants fewer educational materials about reducing salt” → increased

Example input entities:
{entities}

Example input title and abstract:
""
Title: "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion"

Abstract: "Context: Extrapolations from observational studies and short-term intervention trials suggest that population-wide moderation of salt intake might reduce cardiovascular events. Objective: To assess whether 24-hour urinary sodium excretion predicts blood pressure (BP) and health outcomes. Design, setting, and participants: Prospective population study, involving 3681 participants without cardiovascular disease (CVD) who are members of families that were randomly enrolled in the Flemish Study on Genes, Environment, and Health Outcomes (1985-2004) or in the European Project on Genes in Hypertension (1999-2001). Of 3681 participants without CVD, 2096 were normotensive at baseline and 1499 had BP and sodium excretion measured at baseline and last follow-up (2005-2008). Main outcome measures: Incidence of mortality and morbidity and association between changes in BP and sodium excretion. Multivariable-adjusted hazard ratios (HRs) express the risk in tertiles of sodium excretion relative to average risk in the whole study population. Results: Among 3681 participants followed up for a median 7.9 years, CVD deaths decreased across increasing tertiles of 24-hour sodium excretion, from 50 deaths in the low (mean, 107 mmol), 24 in the medium (mean, 168 mmol), and 10 in the high excretion group (mean, 260 mmol; P < .001), resulting in respective death rates of 4.1% (95% confidence interval [CI], 3.5%-4.7%), 1.9% (95% CI, 1.5%-2.3%), and 0.8% (95% CI, 0.5%-1.1%). In multivariable-adjusted analyses, this inverse association retained significance (P = .02): the HR in the low tertile was 1.56 (95% CI, 1.02-2.36; P = .04). Baseline sodium excretion predicted neither total mortality (P = .10) nor fatal combined with nonfatal CVD events (P = .55). Among 2096 participants followed up for 6.5 years, the risk of hypertension did not increase across increasing tertiles (P = .93). Incident hypertension was 187 (27.0%; HR, 1.00; 95% CI, 0.87-1.16) in the low, 190 (26.6%; HR, 1.02; 95% CI, 0.89-1.16) in the medium, and 175 (25.4%; HR, 0.98; 95% CI, 0.86-1.12) in the high sodium excretion group. In 1499 participants followed up for 6.1 years, systolic blood pressure increased by 0.37 mm Hg per year (P < .001), whereas sodium excretion did not change (-0.45 mmol per year, P = .15). However, in multivariable-adjusted analyses, a 100-mmol increase in sodium excretion was associated with 1.71 mm Hg increase in systolic blood pressure (P.<001) but no change in diastolic BP. Conclusions: In this population-based cohort, systolic blood pressure, but not diastolic pressure, changes over time aligned with change in sodium excretion, but this association did not translate into a higher risk of hypertension or CVD complications. Lower sodium excretion was associated with higher CVD mortality.
""

Example output:
[
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “systolic blood pressure changes”,
        “significance”: “positive”,
        “direction”: “increase”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “diastolic blood pressure changes”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “no_change”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “incidence of hypertension”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “low sodium excretion group”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 2096
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “CVD mortality”,
        “significance”: “positive”,
        “direction”: “decrease”,
        “population_main_condition”: “not found”,
        “comparator”: “low sodium excretion group”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “total mortality”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “fatal and nonfatal cardiovascular disease events”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }}
]

Only output in JSON format, do not include any explanations or additional text in your response.
"""

prompt_template_matching = """I extracted exposures and outcomes from abstracts of studies.
I will send you extracted exposure and extracted outcome in a dictionary.

please help me

1. classify if the extracted exposure concept is 'salt/urinary sodium/sodium chloride/NaCI' or synonyms (must be same or highly associated concept)
2. classify if the extracted outcome concept is 'cardiovascular events/diseases, cvd events/diseases, any kind of mortality or death' or synonyms (must be same or highly associated concept)

note CVD events are major health incidents or outcomes that stem from cardiovascular disease (CVD)—that is, disease of the heart and blood vessels. Common examples include: •  Heart attack (myocardial infarction) • Stroke (cerebrovascular accident) •    Unstable angina •  Sudden cardiac death • Heart failure exacerbation •   Coronary revascularization procedures (e.g., bypass surgery, angioplasty)

this is the input and output strucuture:
input: {{extracted_exposure: ''; extracted_outcome: ''}}
output: {{exposure_match: 'yes'/'no', outcome_match: 'yes'/'no'}}

below are examples
example 1:
input: {{extracted_exposure: 'high salt intake'; extracted_outcome: 'blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 2:
input: {{extracted_exposure: 'reduced dietary salt'; extracted_outcome: 'diastolic blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 3:
input: {{extracted_exposure: 'exercise'; extracted_outcome: 'myocardial infarction'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 4:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'stroke'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 5:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'all cause mortality'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

only return the output json, do not output other descriptive words"""

# =======================
# Globals / locks
# =======================

total_input_tokens = 0
total_output_tokens = 0
token_lock = threading.Lock()


# =======================
# Helpers
# =======================

def _strip_code_fences(s: str) -> str:
    """Removes markdown code fences from a string."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip("\n")
    return s.strip()


def _replace_smart_quotes(s: str) -> str:
    """Replaces smart quotes with standard double quotes."""
    if not isinstance(s, str):
        return s
    return (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )


def _parse_json_result(s: str):
    """
    Safely parses a string into a JSON object.
    Tries json.loads first, then falls back to literal_eval.
    """
    s = _strip_code_fences(s)
    s = _replace_smart_quotes(s)
    try:
        return json.loads(s)
    except Exception:
        try:
            return literal_eval(s)
        except Exception as e:
            raise ValueError(f"Could not parse model output as JSON. Original error: {e}\nOutput: {s}")


# =======================
# Core LLM steps
# =======================

def two_step_extraction(text: str, client: OpenAI, model: str):
    """
    Performs the full two-step extraction for a single text entry.
    Returns the parsed JSON object directly.
    """
    global total_input_tokens, total_output_tokens

    # Step 1: Extract entities
    completion_step1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an evidence-based medicine annotator, particularly in effect of salt or urinary sodium on cardiovascular events, help me extract structured information from free texts of article titles and abstracts"},
            {"role": "user", "content": prompt_step1},
            {"role": "user", "content": text}
        ]
    )

    try:
        step1_tokens_in = completion_step1.usage.prompt_tokens
        step1_tokens_out = completion_step1.usage.completion_tokens
    except Exception:
        step1_tokens_in = step1_tokens_out = 0

    with token_lock:
        total_input_tokens += step1_tokens_in
        total_output_tokens += step1_tokens_out

    entities_result = completion_step1.choices[0].message.content or ""
    entities_dict = _parse_json_result(entities_result)
    entities_json = json.dumps(entities_dict, indent=2)

    # Step 2: Extract associations based on entities
    prompt_step2 = prompt_step2_template.format(entities=entities_json)

    completion_step2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an evidence-based medicine annotator, particularly in effect of salt or urinary sodium on cardiovascular events, help me infer associations from given entities and titles and abstracts"},
            {"role": "user", "content": prompt_step2},
            {"role": "user", "content": f"extracted entities: {entities_json}"},
            {"role": "user", "content": f"title and abstract: {text}"}
        ]
    )

    try:
        step2_tokens_in = completion_step2.usage.prompt_tokens
        step2_tokens_out = completion_step2.usage.completion_tokens
    except Exception:
        step2_tokens_in = step2_tokens_out = 0

    with token_lock:
        total_input_tokens += step2_tokens_in
        total_output_tokens += step2_tokens_out

    associations_result = completion_step2.choices[0].message.content or ""
    return _parse_json_result(associations_result)


def matching_pair(text: str, prompt: str, client: OpenAI, model: str):
    """
    Classifies an extracted exposure/outcome pair against target concepts.
    Returns the parsed JSON object directly.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an expert in epidemiology, specialized in social determinants of health (SDoh), especially in effect of salt on cardiovascular events and mortality. "},
            {"role": "user", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    result = completion.choices[0].message.content or "{}"
    return _parse_json_result(result)


# =======================
# Multithreaded runners
# =======================

def _worker_extract(input_queue, output_queue, progress_bar, client, model):
    """Worker function for two-step extraction."""
    while True:
        item = input_queue.get()
        if item is None:
            break
        pmid, text = item
        try:
            extracted_results = two_step_extraction(text, client, model)
            for res in extracted_results:
                res['pmid'] = pmid
                output_queue.put(res)
        except Exception as e:
            print(f"Error processing PMID {pmid}: {e}", file=sys.stderr)
            try:
                preview = str(text)[:500]
            except Exception:
                preview = "N/A"
            print(f"Input text: {preview}...", file=sys.stderr)
            if 'extracted_results' in locals() and extracted_results is not None:
                print(f"Failed output: {extracted_results}", file=sys.stderr)
        finally:
            input_queue.task_done()
            progress_bar.update(1)


def _worker_match(input_queue, output_queue, progress_bar, client, model, prompt_template_matching):
    """Worker function for concept matching."""
    while True:
        item = input_queue.get()
        if item is None:
            break
        index, row = item
        try:
            result = matching_pair(row['classify_input'], prompt_template_matching, client, model)
            output_queue.put((index, result))
        except Exception as e:
            print(f"Error processing matching for row {index}: {e}", file=sys.stderr)
        finally:
            input_queue.task_done()
            progress_bar.update(1)


def process_file(file_path: str, client: OpenAI, model: str) -> Optional[pd.DataFrame]:
    """
    Reads data from an XLSX or CSV file and runs the extraction pipeline.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Error: File not found at {file_path}", file=sys.stderr)
        return None

    file_extension = Path(file_path).suffix.lower()
    df = None
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        print(f"⚠️ Error: Unsupported file type '{file_extension}'. Only .csv and .xlsx are supported.",
              file=sys.stderr)
        return None

    if df.empty:
        print(f"⚠️ Warning: The file {file_path} is empty.", file=sys.stderr)
        return None

    return run_extraction(df, client, model)


def run_extraction(all_got_df: pd.DataFrame,
                   client: OpenAI,
                   model: str,
                   max_workers: int = 4) -> pd.DataFrame:
    """
    Orchestrates LLM extraction and matching to produce a structured DataFrame.
    """
    if all_got_df is None or all_got_df.empty:
        raise ValueError("all_got_df is empty or None.")

    df_in = all_got_df.copy()
    for col in ['title', 'abstract']:
        if col not in df_in.columns:
            df_in[col] = ""
    df_in['text'] = "Title: " + df_in['title'].fillna("") + "\n\nAbstract: " + df_in['abstract'].fillna("")

    if 'pmid' not in df_in.columns:
        # If no pmid, create a unique identifier for each row
        df_in['pmid'] = range(len(df_in))

    global total_input_tokens, total_output_tokens
    total_input_tokens = 0
    total_output_tokens = 0

    results = []
    input_queue = Queue()
    output_queue = Queue()

    progress_bar = tqdm(total=df_in.shape[0], desc=f"Extracting with {model}", position=0, leave=True)

    threads = []
    for _ in range(max_workers):
        t = threading.Thread(
            target=_worker_extract,
            args=(input_queue, output_queue, progress_bar, client, model),
            daemon=True
        )
        t.start()
        threads.append(t)

    for _, row in df_in.iterrows():
        input_queue.put((row['pmid'], row['text']))

    input_queue.join()

    for _ in threads:
        input_queue.put(None)
    for t in threads:
        t.join()

    progress_bar.close()

    while not output_queue.empty():
        results.append(output_queue.get())

    print(f"Total input tokens for {model}: {total_input_tokens}")
    print(f"Total output tokens for {model}: {total_output_tokens}")
    print(f"Total tokens used (input + output) for {model}: {total_input_tokens + total_output_tokens}")

    if not results:
        print("No results extracted. Returning original DataFrame.")
        return df_in[['pmid']].copy()

    extracted_df = pd.json_normalize(results)

    # Now perform the matching step
    extracted_df['classify_input'] = extracted_df.apply(
        lambda r: json.dumps({
            "extracted_exposure": r.get('exposure', ''),
            "extracted_outcome": r.get('outcome', '')
        }),
        axis=1
    )

    matched_df = match_results(extracted_df, client=client, model=model)

    # Add pub_year from original DataFrame and ensure proper typing
    year_df = df_in[['pmid', 'pub_year']].drop_duplicates() if 'pub_year' in df_in.columns else df_in[
        ['pmid']].drop_duplicates()
    final_df = matched_df.merge(year_df, on='pmid', how='left')
    if 'pub_year' in final_df.columns:
        final_df['pub_year'] = pd.to_numeric(final_df['pub_year'], errors='coerce')

    return final_df


def match_results(df: pd.DataFrame, client: OpenAI, model: str) -> pd.DataFrame:
    """
    Matches extracted exposures and outcomes against target concepts using threading.
    """
    input_queue = Queue()
    output_queue = Queue()

    progress_bar = tqdm(total=df.shape[0], desc=f"Matching with {model}", position=0, leave=True)

    num_workers = 8
    threads = []
    # Create the prompt once outside the loop for efficiency
    prompt_template_matching = """I extracted exposures and outcomes from abstracts of studies.
I will send you extracted exposure and extracted outcome in a dictionary.

please help me

1. classify if the extracted exposure concept is 'salt/urinary sodium/sodium chloride/NaCI' or synonyms (must be same or highly associated concept)
2. classify if the extracted outcome concept is 'cardiovascular events/diseases, cvd events/diseases, any kind of mortality or death' or synonyms (must be same or highly associated concept)

note CVD events are major health incidents or outcomes that stem from cardiovascular disease (CVD)—that is, disease of the heart and blood vessels. Common examples include: •  Heart attack (myocardial infarction) • Stroke (cerebrovascular accident) •    Unstable angina •  Sudden cardiac death • Heart failure exacerbation •   Coronary revascularization procedures (e.g., bypass surgery, angioplasty)

this is the input and output strucuture:
input: {{extracted_exposure: ''; extracted_outcome: ''}}
output: {{exposure_match: 'yes'/'no', outcome_match: 'yes'/'no'}}

below are examples
example 1:
input: {{extracted_exposure: 'high salt intake'; extracted_outcome: 'blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 2:
input: {{extracted_exposure: 'reduced dietary salt'; extracted_outcome: 'diastolic blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 3:
input: {{extracted_exposure: 'exercise'; extracted_outcome: 'myocardial infarction'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 4:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'stroke'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 5:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'all cause mortality'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

only return the output json, do not output other descriptive words"""

    for _ in range(num_workers):
        t = threading.Thread(
            target=_worker_match,
            args=(input_queue, output_queue, progress_bar, client, model, prompt_template_matching),
            daemon=True
        )
        t.start()
        threads.append(t)

    for index, row in df.iterrows():
        input_queue.put((index, row))

    input_queue.join()

    for _ in threads:
        input_queue.put(None)
    for t in threads:
        t.join()

    progress_bar.close()

    while not output_queue.empty():
        index, result = output_queue.get()
        df.loc[index, 'exposure_match'] = result.get('exposure_match', 'NA')
        df.loc[index, 'outcome_match'] = result.get('outcome_match', 'NA')

    return df


# Deprecated - use run_extraction instead
def compile_extracted_results(input_df: pd.DataFrame, text_column: str, client: OpenAI, model: str) -> pd.DataFrame:
    """
    DEPRECATED. Please use the `run_extraction` function instead.
    """
    print(
        "WARNING: `compile_extracted_results` is deprecated. Use `run_extraction` for a more robust and complete pipeline.")
    return run_extraction(input_df.rename(columns={text_column: 'text', 'pmid': 'pmid'}), client, model)
