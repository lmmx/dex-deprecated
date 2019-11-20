from pathlib import Path
import configparser as cp

top_dir = Path(__file__).absolute().parent.parent
shelves_dir = top_dir / "shelves"
auths = [x.name for x in shelves_dir.iterdir()]

def get_coauth_name(coauths_dir, auth):
    ccf = cp.ConfigParser()
    coauth_info = coauths_dir / auth / 'auth_info.ini'
    assert coauth_info.exists(), f"No auth_info.ini file for {auth}"
    assert ccf.read(coauth_info), f"Could not read {coauth_info}"
    first, last = ccf['NAME']['first_names'], ccf['NAME']['surname'] 
    return first, last

def book_lookup(auth,year):
    """Get book info given its author and publication year."""
    bcf = cp.ConfigParser()
    book_info = shelves_dir / auth / year / 'book_info.ini'
    assert book_info.exists(), f"No book_info.ini file for {auth}/{year}"
    assert bcf.read(book_info), f"Could not read {book_info}"
    title = bcf['BIB']['title']
    if 'edition' in bcf['BIB']:
        edition = bcf['BIB']['edition']
    else:
        edition = None
    if 'coauth_keys' in bcf['INTERNAL']:
        coauths = [c.strip() for c in bcf['INTERNAL']['coauth_keys'].split(',')]
    else:
        coauths = None
    return title, edition, coauths

def make_author_entry(auth):
    """Create a summary string for a given author"""
    auth_info = shelves_dir / auth / 'auth_info.ini'
    if auth_info.exists():
        acf = cp.ConfigParser()
        acf.read(auth_info)
        first, last = acf['NAME']['first_names'], acf['NAME']['surname']
        book_anchor = f"#book-{auth}".replace('_','-')
        auth_entry = [f"\n### [:book: {last}, {first}]({book_anchor}):"]
    else:
        auth_entry = [f"\n- {auth}"]
    auth_year_list = [x.name for x in (shelves_dir / auth).iterdir() if x.is_dir()]
    for year in sorted(auth_year_list):
        title, edition, coauths = book_lookup(auth, year)
        auth_str = f"  - {year} — *{title}*"
        if edition is not None: auth_str += f" ({edition}e.)"
        if coauths is not None:
            coauth_dir = shelves_dir / auth / year / 'coauths'
            coauth_surname_list = [get_coauth_name(coauth_dir, c)[1] for c in coauths]
            coauth_surnames = ', '.join(coauth_surname_list[:-1])
            if len(coauth_surname_list) > 2:
                coauth_surnames += ',' # Oxford comma
            if len(coauth_surname_list) > 1:
                coauth_surnames += ' & '
            coauth_surnames += coauth_surname_list[-1]
            auth_str += (f" with {coauth_surnames}")
        auth_entry.append(auth_str)
    auth_entry.append('')
    return '\n'.join(auth_entry)

all_auth_entries = [make_author_entry(author) for author in auths]
new_lines = [auth_str for auth_str in all_auth_entries]
README = top_dir / "README.md"
with open(README) as f:
    ll = [l.rstrip() for l in f.readlines()]
    AW_str = "[//]: # (AUTOWRITE::"
    start, end = [i for i, x in enumerate([l.startswith(AW_str) for l in ll]) if x]

new_README = '\n'.join(ll[0:start+1] + new_lines + ll[end:])

with open(README, 'w+') as f:
    f.write(new_README)
