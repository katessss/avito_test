"""Нормализация текста: нижний регистр, ё→е, токенизация и лемматизация (pymorphy3).

Лемматизация нужна, потому что запросы и объявления пишут в разных формах:
«скупка телевизоров» ↔ «Скупаем телевизор». Леммы кешируются по словам —
уникальных слов немного, поэтому весь корпус обрабатывается за ~2 минуты.
"""
import re

import pymorphy3

_TOKEN_RE = re.compile(r'[a-zа-я0-9]+')
_CYR_RE = re.compile('[а-я]')
_morph = None
_cache = {}


def _lemma(word):
    global _morph
    lemma = _cache.get(word)
    if lemma is None:
        # цифры, латиница и однобуквенные токены оставляем как есть
        if word.isdigit() or len(word) < 2 or not _CYR_RE.search(word):
            lemma = word
        else:
            if _morph is None:
                _morph = pymorphy3.MorphAnalyzer()
            lemma = _morph.parse(word)[0].normal_form.replace('ё', 'е')
        _cache[word] = lemma
    return lemma


def normalize(text, max_chars=None):
    """Строка → строка лемм через пробел."""
    if text is None or isinstance(text, float):
        return ''
    text = text.lower().replace('ё', 'е')
    if max_chars:
        text = text[:max_chars]
    return ' '.join(_lemma(w) for w in _TOKEN_RE.findall(text))


def raw_lower(text):
    """Мягкая нормализация без лемматизации — для символьных n-грамм."""
    return (text or '').lower().replace('ё', 'е')


# В item_infm_params_text адрес («Место оказания услуг Москва, ул. ...») идёт
# вперемешку с видом/типом услуги. Адрес шумит в текстовом поиске, поэтому
# выделяем его в отдельное поле.
_ADDR_RE = re.compile(
    r'Место оказания услуг (.*?)(?= Тип стоимости| Тип услуги| Вид услуги| Куда выезжаете'
    r'| Гарантия| Опыт работы| Где вы| Дополнительно|$)')


def split_params(params):
    """'Вид услуги X Место оказания услуг <адрес> Тип ...' → (параметры без адреса, адрес)."""
    params = params or ''
    m = _ADDR_RE.search(params)
    if not m:
        return params, ''
    addr = m.group(1)
    return params.replace('Место оказания услуг ' + addr, ' '), addr
