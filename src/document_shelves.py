from pathlib import Path
import configparser as cp

top_dir = Path(__file__).absolute().parent.parent
shelves_dir = top_dir / "shelves"
auths = [x.name for x in shelves_dir.iterdir()]

def book_lookup(auth,year):
    """Get the title of a book given its author and publication year."""
    bcf = cp.ConfigParser()
    book_info = shelves_dir / auth / year / 'book_info.ini'
    assert book_info.exists(), f"No book_info.ini file for {auth}/{year}"
    assert bcf.read(book_info), f"Could not read {book_info}"
    title = bcf['BIB']['title']
    if 'edition' in bcf['BIB']:
        edition = bcf['BIB']['edition']
    else:
        edition = None
    return title, edition

def make_author_entry(auth):
    """Create a summary string for a given author"""
    auth_info = shelves_dir / auth / 'auth_info.ini'
    if auth_info.exists():
        acf = cp.ConfigParser()
        acf.read(auth_info)
        first, last = acf['NAME']['first_names'], acf['NAME']['surname']
        auth_entry = [f"\n- {last}, {first}:"]
    else:
        auth_entry = [f"\n- {auth}"]
    auth_year_list = [x.name for x in (shelves_dir / auth).iterdir() if x.is_dir()]
    for n, year in enumerate(sorted(auth_year_list)):
        title, edition = book_lookup(auth, year)
        if edition is not None:
            auth_entry.append(f"  {n+1}) {year} — {title} ({edition}e.)")
        else:
            auth_entry.append(f"  {n+1}) {year} — {title}")
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
