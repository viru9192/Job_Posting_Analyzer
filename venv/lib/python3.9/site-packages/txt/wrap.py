from textwrap import wrap as _wrap
from .align import align as align_text


def wrap(
		text,
		max_width,
		align='left',
		indentation='',
		prefix='',
		suffix='',
		left_margin=0,
		right_margin=0
):
	remaining_width = max_width - len(indentation) - len(prefix) - len(suffix) - left_margin - right_margin
	inside_width = max_width - len(prefix) - len(suffix)

	return '\n'.join([
		prefix + align_text(
			text=f'{" " * left_margin}{indentation}{line}{" " * right_margin}',
			width=inside_width,
			side=align
		) + suffix
		for line in _wrap(text=text, width=remaining_width)
	])
