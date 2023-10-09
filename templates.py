
# srcs/streamlit_app/templates.py
# import urllib.parse

def load_css() -> str:
    """ Return all css styles. """
    common_tag_css = """
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: .15rem .40rem;
                position: relative;
                text-decoration: none;
                font-size: 95%;
                border-radius: 5px;
                margin-right: .5rem;
                margin-top: .4rem;
                margin-bottom: .5rem;
    """
    return f"""
        <style>
            #tags {{
                {common_tag_css}
                color: rgb(88, 88, 88);
                border-width: 0px;
                background-color: rgb(240, 242, 246);
            }}
            #tags:hover {{
                color: black;
                box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
            }}
            #active-tag {{
                {common_tag_css}
                color: rgb(246, 51, 102);
                border-width: 1px;
                border-style: solid;
                border-color: rgb(246, 51, 102);
            }}
            #active-tag:hover {{
                color: black;
                border-color: black;
                background-color: rgb(240, 242, 246);
                box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
            }}
        </style>
    """

def number_of_results(total_hits: int, duration: float) -> str:
    """ HTML scripts to display number of results and duration. """
    return f"""
        <div style="color:grey;font-size:95%;">
            {total_hits} results ({duration:.2f} seconds)
        </div><br>
    """

def search_result(i: int, url: str, title: str, content: str,
                  length: str, episode_num: int, **kwargs) -> str:
    """ HTML scripts to display search results. """
    return f"""
        <div style="font-size:120%;">
            {i + 1}.
            <a href="{url}">
                {title}
            </a>
        </div>
        <div style="font-size:95%;">
            <div style="color:grey;float:left;">
                ...
            </div>
            {content}
            <p><b>Episode length: {length}</b><br><b>Episode # {episode_num}</b></p>
        </div>
    """
            # <div style="color:grey;font-size:95%;">
            #     {url[:90] + '...' if len(url) > 100 else url}
            # </div>
            
# def tag_boxes(search: str, tags: list, active_tag: str) -> str:
#     """ HTML scripts to render tag boxes. """
#     html = ''
#     search = urllib.parse.quote(search)
#     for tag in tags:
#         if tag != active_tag:
#             html += f"""
#             <a id="tags" href="?search={search}&tags={tag}">
#                 {tag.replace('-', ' ')}
#             </a>
#             """
#         else:
#             html += f"""
#             <a id="active-tag" href="?search={search}">
#                 {tag.replace('-', ' ')}
#             </a>
#             """

#     html += '<br><br>'
#     return html
