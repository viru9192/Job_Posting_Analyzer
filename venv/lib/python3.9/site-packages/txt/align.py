def align(text, width, side='left'):
	if side.lower()[0] == 'c':
		return text.center(width)
	elif side.lower()[0] == 'r':
		return text.rjust(width)
	else:
		return text.ljust(width)
