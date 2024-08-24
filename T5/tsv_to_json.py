import csv
import json


def tsv_to_json(tsv_file, json_file):
    data = []

    # Reading the TSV file
    with open(tsv_file, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.DictReader(file, delimiter='\t')

        # Convert each row into a dictionary and add it to the list
        for row in tsv_reader:
            data.append({
                "text": row["string1"] + " sentence2:" + row["string2"],
                "label": int(row["Label"])
            })

    # Writing the list of dictionaries to a JSON file
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Example usage
tsv_file = '.\dataset\Rel_train.tsv'  # Replace with your TSV file path
json_file = '.\dataset\Rel_train.json'  # Replace with your desired JSON output file path

#tsv_file = '.\dataset\Rel_dev_new.tsv'  # Replace with your TSV file path
#json_file = '.\dataset\Rel_dev_new.json'
#tsv_file = '.\dataset\Rel_test.tsv'  # Replace with your TSV file path
#json_file = '.\dataset\Rel_test.json'

tsv_to_json(tsv_file, json_file)
