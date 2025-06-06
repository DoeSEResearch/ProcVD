import pandas as pd
import json
import logging
import argparse
from extract_vulnerability_lines import extract_vulnerability_lines

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_data(test_limit=None):
    """
    This function processes the Excel data, filters the required columns,
    and generates a JSON file with additional processing.
    """
    # Fixed input and output file paths
    file_path = "../data/cvefixes_filter_data.xlsx"
    output_path = "../data/processed_cvefixes.json"

    logging.info(f"Start reading dataset from: {file_path}")
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        total_records = len(df)
        logging.info(f"Successfully read {total_records} records from the dataset.")
    except Exception as e:
        logging.error(f"Failed to read the Excel file: {e}")
        return

    # Rename columns
    df = df.rename(columns={
        'description': 'cve_description',  # Rename description column to cve_description
        'before_change': 'label'  # Rename before_change column to label
    })

    # Keep only the relevant columns
    keep_fields = ['published_date', 'cve_id', 'cwe_id', 'code', 'programming_language', 'cve_description', 'label',
                   'vulnerability_diff', 'start_line', 'end_line']
    df = df[keep_fields]

    # Convert label to 0 and 1
    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

    # Limit data processing if 'limit' is provided
    if test_limit:
        logging.info(f"Test mode, processing only the first {test_limit} records")
        df = df.head(test_limit)

    # Initialize result list to store the processed data
    result = []
    discarded_count = 0  # Track discarded records
    logging.info("Processing each record...")

    # Iterate over each record
    for idx, row in df.iterrows():
        item = {
            "published_date": row['published_date'],
            "cve_id": row['cve_id'],
            "cwe_id": row['cwe_id'],
            "code": row['code'],
            "programming_language": row['programming_language'],
            "label": row['label'],
        }

        # Process the cve_description to only keep the actual description value
        cve_description = eval(row['cve_description'])[0]  # Parsing the string to get the dictionary
        item["cve_description"] = cve_description['value']  # Keep only the description text

        # Use the extract_vulnerability_lines function to extract vulnerability lines and fix lines
        vuln_diff = extract_vulnerability_lines(row['vulnerability_diff'], row['start_line'], row['end_line'])
        item.update(vuln_diff)  # Add the extracted lines to the item

        # Filter the item based on label: keep vulnerability_line for label=1, fix_line for label=0
        if row['label'] == 1:  # If there is a vulnerability, keep vulnerability_line
            if not item['vulnerability_line']:  # Discard if vulnerability_line is empty
                discarded_count += 1
                logging.info(f"Record {row['cve_id']} has empty vulnerability line and is discarded.")
                continue
            del item['fix_line']  # Remove fix_line if there's a vulnerability line
        else:  # If there's no vulnerability, keep fix_line
            if not item['fix_line']:  # Discard if fix_line is empty
                discarded_count += 1
                logging.info(f"Record {row['cve_id']} has empty fix line and is discarded.")
                continue
            del item['vulnerability_line']  # Remove vulnerability_line if there's a fix line

        # Add the ID to the record, ensuring it is at the beginning of the dictionary
        item = {"id": len(result)} | item  # Put the ID at the start of the dictionary

        # Append the processed item to the result list
        result.append(item)

    # Final valid record count
    final_count = len(result)

    # Save the final result as a JSON file
    if result:
        logging.info(
            f"Processed {total_records} records, discarded {discarded_count} records, and kept {final_count} valid records. Saving to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Data successfully saved to {output_path}")
    else:
        logging.warning("No valid records to save.")


# Main function
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Test mode: process only the first N records")
    args = parser.parse_args()

    # Process data, if limit is passed, process only the first N records
    process_data(test_limit=args.limit)
