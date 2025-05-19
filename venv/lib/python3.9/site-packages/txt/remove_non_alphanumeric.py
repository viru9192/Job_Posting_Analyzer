import re


def remove_non_alphanumeric(s, replace_with='_', keep_underscore=True):
	s = str(s)
	# replace & with and
	s = re.sub(r'&+', '&', s)
	s = s.replace(' & ', ' and ')
	s = s.replace('_&_', '_and_')
	s = s.replace('&', ' and ')

	if keep_underscore:
		return re.sub(r'[\W]+', replace_with, s, flags=re.UNICODE)
	else:
		return re.sub(r'[\W_]+', replace_with, s, flags=re.UNICODE)


def remove_non_alphabetic(s):
	return ''.join([i for i in s if i.isalpha()])
