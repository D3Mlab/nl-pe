def list_to_text_block(texts: list[str], index_str: str = "Doc") -> str:
    """
    Convert a list of texts into a long string where each text is wrapped in
    pseudo-HTML tags. Blocks are separated by double newlines.

    Rules:
    - Texts are indexed starting from 1.
    
    Parameters
    ----------
    texts : list[str]
        List of text items to format.
    index_str : str, optional
        Base name for wrapping tags. Default is "Doc".

    Returns
    -------
    str
        Formatted string.
    """
    rows = []
    for i, text in enumerate(texts, start=1):
        tag_name = f"{index_str}{i}"
        block = f"<{tag_name}>\n{text}\n</{tag_name}>"
        rows.append(block)
    return "\n\n".join(rows)
