import gzip
from Bio import SeqIO
import csv


def get_seq(query_id):
    fasta_file = "9606.protein.sequences.v12.0.fa.gz"
    with gzip.open(fasta_file, "rt") as handle:
        sequences = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

    # Check if the query_id is in the sequences dictionary
    if query_id in sequences:
        return str(sequences[query_id].seq)
    else:
        raise ValueError(f"Query ID '{query_id}' not found in the sequences dictionary.")




def get_word_token():
    file_path = 'word_token.csv'
    result_dict = {}

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) >= 2:  # Ensure at least two columns in a row
                key, value = row[:2]
                result_dict[key] = value

    return result_dict


def tokenization(cell, word_token, n_grams=3):
    if len(cell) / n_grams != 0:
        cell_0 = cell[len(cell) % n_grams:]
    cell_1 = [cell_0[i:i + n_grams] for i in range(0, len(cell_0), n_grams)]
    cell_1 = [word_token[word] for word in cell_1]
    cell = cell_1
    return cell


def pad(cell, fixed_length=1000):
    if len(cell) > fixed_length:
        # If the original list is longer, remove leftmost elements
        return cell[-fixed_length:]
    elif len(cell) < fixed_length:
        # If the original list is shorter, pad leftmost with zeroes
        return [0] * (fixed_length - len(cell)) + cell
    else:
        # If the original list has the desired length, no modifications are needed
        return cell


'''if __name__ == "__main__":
    file_path = "word_token.csv"
    my_dict = read_csv_to_dict(file_path)

    print("Dictionary created from the CSV file:")
    print(my_dict)'''

'''if __name__ == "__main__":
    fasta_file_path = "9606.protein.sequences.v12.0.fa.gz"
    user_query = input("Enter sequence ID to get the sequence: ")

    sequence = get_sequence_from_fasta_gz(fasta_file_path, user_query)

    if sequence is not None:
        print("Sequence:", sequence)
    else:
        print("Sequence not found.")'''
