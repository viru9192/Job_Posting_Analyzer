import re
from .remove_non_alphanumeric import remove_non_alphanumeric as remove_non_alpha

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z0-9])([A-Z])')


def convert_camel_to_snake(string, remove_non_alphanumeric=True):
	"""
	converts CamelCase to snake_case
	:type string: str
	:rtype: str
	"""
	if remove_non_alphanumeric:
		string = remove_non_alpha(string, replace_with='_', keep_underscore=True)

	s1 = _first_cap_re.sub(r'\1_\2', string)
	result = _all_cap_re.sub(r'\1_\2', s1).lower()
	result = re.sub(pattern='\s*_+', repl="_", string=result)
	return result
