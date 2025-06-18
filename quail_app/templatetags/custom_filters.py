from django import template  # BUKAN "templates"

register = template.Library()

@register.filter
def index(sequence, position):
    try:
        return sequence[position]
    except (IndexError, KeyError, TypeError):
        return ''
