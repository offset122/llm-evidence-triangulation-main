import datetime
import time
import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List

__all__ = ['crawler', 'fetch_pubmed']


def pmid_splitter(pmid_list: List[str], chunk_size: int = 300) -> List[List[str]]:
    """Split a large PMID list into chunks of `chunk_size` for NCBI requests."""
    chunks = [pmid_list[i:i + chunk_size] for i in range(0, len(pmid_list), chunk_size)]
    return chunks


def fetch_pubmed(pmid_list: List[str], chunk_size: int = 50) -> pd.DataFrame:
    """Fetch PubMed metadata for a list of PMIDs.

    Parameters
    ----------
    pmid_list : list[str]
        A list of PubMed IDs to retrieve.
    chunk_size : int, default 50
        Number of PMIDs to query per API request to stay within NCBI limits.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing one row per PMID with parsed metadata.
    """
    pmid_chunks = pmid_splitter(pmid_list, chunk_size)
    return crawler(pmid_chunks)


def crawler(pmid_chunks: List[List[str]]) -> pd.DataFrame:
    """
    Crawls PubMed to fetch article metadata in chunks.

    Parameters
    ----------
    pmid_chunks : list[list[str]]
        A list of lists, where each inner list is a chunk of PMIDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with all the parsed PubMed data.
    """
    full_df_list = []
    count = 0
    retmode = "xml"
    rettype = ""

    # Optional NCBI identification
    ncbi_params = {}
    ncbi_email = os.getenv("NCBI_EMAIL")
    if ncbi_email:
        ncbi_params["email"] = ncbi_email
    ncbi_api_key = os.getenv("NCBI_API_KEY")
    if ncbi_api_key:
        ncbi_params["api_key"] = ncbi_api_key

    for chunk in pmid_chunks:
        print(f'chunk #{count} starts at: {datetime.datetime.now()}')

        chunk_str = ",".join(chunk)

        try:
            params = {
                    "db": "pubmed",
                    "retmode": retmode,
                    "id": chunk_str,
                    "rettype": rettype,
                }
            params.update(ncbi_params)
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        except requests.exceptions.RequestException as e:
            print(f"Request failed for chunk #{count}: {e} – skipping")
            count += 1
            time.sleep(5)
            continue

        try:
            root = ET.fromstring(resp.text.encode("utf-8"))
        except ET.ParseError as e:
            print(f"XML parse error for chunk #{count}: {e} – skipping")
            count += 1
            time.sleep(5)
            continue

        chunk_data = {}
        for article in root.findall('PubmedArticle/MedlineCitation'):
            pmid = article.find('PMID').text
            info_dict = {}

            # --- Article Info ---
            article_tree = article.find('Article')
            if article_tree is not None:
                # Publication Date
                pub_date = article_tree.find('.//PubDate')
                year = pub_date.find('Year').text if pub_date is not None and pub_date.find(
                    'Year') is not None else None
                month = pub_date.find('Month').text if pub_date is not None and pub_date.find(
                    'Month') is not None else None

                # Electronic Publication Date
                epub_date = article_tree.find('.//ArticleDate[@DateType="Electronic"]')
                epub_year = epub_date.find('Year').text if epub_date is not None and epub_date.find(
                    'Year') is not None else None
                epub_month = epub_date.find('Month').text if epub_date is not None and epub_date.find(
                    'Month') is not None else None

                info_dict['pub_year'] = year
                info_dict['pub_month'] = month
                info_dict['epub_year'] = epub_year
                info_dict['epub_month'] = epub_month

                # Publication Types
                publication_types = [pub_type.text for pub_type in article_tree.findall('.//PublicationType')]
                info_dict['pub_types'] = publication_types

                # Article Title
                article_title = article_tree.find('.//ArticleTitle').text if article_tree.find(
                    './/ArticleTitle') is not None else None
                info_dict['article_title'] = article_title

                # Abstract
                abstract_text_elements = article_tree.findall('.//AbstractText')
                full_abs = "".join([i.text for i in abstract_text_elements if i.text is not None])
                info_dict['full_abs'] = full_abs

                background, conclusion, result, method, label_conclusion, label_result = None, None, None, None, None, None
                for abs_part in abstract_text_elements:
                    if 'NlmCategory' in abs_part.attrib:
                        category = abs_part.attrib['NlmCategory']
                        text = abs_part.text
                        if category == 'BACKGROUND':
                            background = text
                        elif category == 'CONCLUSIONS':
                            conclusion = text
                        elif category == 'RESULTS':
                            result = text
                        elif category == 'METHODS':
                            method = text

                    if 'Label' in abs_part.attrib:
                        label = abs_part.attrib['Label'].lower()
                        text = abs_part.text
                        if 'conclusion' in label:
                            label_conclusion = text
                        elif 'result' in label:
                            label_result = text

                info_dict['background'] = background
                info_dict['method'] = method
                info_dict['result'] = result
                info_dict['conclusion'] = conclusion
                info_dict['label_result'] = label_result
                info_dict['label_conclusion'] = label_conclusion

            # --- Commentaries Info ---
            comment_tree = article.find('CommentsCorrectionsList')
            comment_count = sum(
                1 for i in comment_tree if i.attrib.get('RefType') == 'CommentIn') if comment_tree is not None else 0
            info_dict['# of comments'] = comment_count

            # --- Mesh Info ---
            mesh_tree = article.find('MeshHeadingList')
            all_mesh_list = []
            major_mesh_list = []
            if mesh_tree is not None:
                for mesh_heading in mesh_tree.findall('MeshHeading'):
                    full_mesh_parts = []
                    major_mesh_parts = []

                    descriptor = mesh_heading.find('DescriptorName')
                    if descriptor is not None:
                        full_mesh_parts.append(descriptor.text)
                        if descriptor.attrib.get('MajorTopicYN') == 'Y':
                            major_mesh_parts.append(descriptor.text + '*')

                    for qualifier in mesh_heading.findall('QualifierName'):
                        full_mesh_parts.append(qualifier.text)
                        if qualifier.attrib.get('MajorTopicYN') == 'Y':
                            major_mesh_parts.append(qualifier.text + '*')

                    all_mesh_list.append("/".join(full_mesh_parts))
                    if major_mesh_parts:
                        major_mesh_list.append("/".join(major_mesh_parts))

            info_dict['all_mesh_terms'] = all_mesh_list
            info_dict['major_mesh_terms'] = major_mesh_list

            chunk_data[pmid] = info_dict

        if chunk_data:
            info_df = pd.DataFrame.from_dict(chunk_data, orient='index').reset_index().rename(columns={'index': 'pmid'})
            full_df_list.append(info_df)

        print(f'chunk #{count} is done at: {datetime.datetime.now()}')
        count += 1
        print('\n')
        time.sleep(5)

    if not full_df_list:
        return pd.DataFrame()

    final_df = pd.concat(full_df_list, ignore_index=True)
    return final_df
