import os
def load_five_words_poerties(data_path):
	input_file = os.path.join(data_path)
	poetries = []
	with open(input_file, "r", encoding='utf-8') as f:
		for line in f:
			if len(line) == 25:
				poetries.append(line)

	output_file = os.path.join("./data/five_words_poetries.txt")
	with open(output_file, "w", encoding='utf-8') as f:
		f.write("".join(poetries))


load_five_words_poerties("./data/poetry_data.txt")
