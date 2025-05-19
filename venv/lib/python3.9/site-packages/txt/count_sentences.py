ENDINGS = ['.', '!', '?', ';']


def count_sentences(text):
	text = text.strip()
	if len(text) == 0:
		return 0

	split_result = None
	for ending in ENDINGS:
		separator = f'{ending} '

		if split_result is None:
			split_result = text.split(separator)

		else:
			split_result = [y for x in split_result for y in x.split(separator)]

	last_is_sentence = text[-1] in ENDINGS

	return len(split_result) - 1 + last_is_sentence
