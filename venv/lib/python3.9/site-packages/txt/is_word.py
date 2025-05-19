def is_word(s):
	if not isinstance(s, str):
		raise TypeError(f'{s} if of type {type(s)}')

	return any([character.isalpha() for character in s])


def count_words(text):
	if not isinstance(text, str):
		raise TypeError(f'{text} if of type {type(text)}')

	return len([s for s in text.split() if is_word(s)])
