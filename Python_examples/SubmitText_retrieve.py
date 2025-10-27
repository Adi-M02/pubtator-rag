import requests
import os
import sys

def submit_text_retrieve(input_folder, inputfile_session_number, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sn_hash = {}
    with open(inputfile_session_number, 'r') as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                session_number, input_filename = line.split("\t")
                sn_hash[session_number] = input_filename
    
    base_url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi"
    
    for session_number, input_filename in sn_hash.items():
        output_file_path = os.path.join(output_folder, input_filename)
        if os.path.exists(output_file_path):
            print(f"{output_file_path} - finished")
            continue

        retrieve_url = f"{base_url}?id={session_number}"
        response = requests.get(retrieve_url)
        if response.status_code == 200:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(response.text)
            print(f"{session_number} : Result is retrieved.")
        else:
            print(f"{session_number} : Error retrieving results - {response.status_line}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nUsage: python submit_text_retrieve.py [InputFolder] [InputFileSessionNumber] [OutputFolder]\n")
    else:
        input_folder = sys.argv[1]
        inputfile_session_number = sys.argv[2]
        output_folder = sys.argv[3]
        submit_text_retrieve(input_folder, inputfile_session_number, output_folder)
