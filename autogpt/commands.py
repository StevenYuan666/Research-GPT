import json
import re
import PyPDF2
from scholarly import scholarly
from typing import List
from arxiv import Search, SortCriterion, SortOrder
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, Cohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS, VectorStore
import datetime
import autogpt.agent_manager as agents
from autogpt.config import Config
from autogpt.json_parser import fix_and_parse_json
from autogpt.image_gen import generate_image
from duckduckgo_search import ddg
from autogpt.ai_functions import evaluate_code, improve_code, write_tests
from autogpt.browse import scrape_links, scrape_text, summarize_text
from autogpt.execute_code import execute_python_file, execute_shell
from autogpt.file_operations import (
    append_to_file,
    delete_file,
    read_file,
    search_files,
    write_to_file,
)
from autogpt.memory import get_memory
from autogpt.speak import say_text
from autogpt.web import browse_website


cfg = Config()


def is_valid_int(value) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def get_command(response):
    """Parse the response and return the command name and arguments"""
    try:
        response_json = fix_and_parse_json(response)

        if "command" not in response_json:
            return "Error:", "Missing 'command' object in JSON"

        if not isinstance(response_json, dict):
            return "Error:", f"'response_json' object is not dictionary {response_json}"

        command = response_json["command"]
        if not isinstance(command, dict):
            return "Error:", "'command' object is not a dictionary"

        if "name" not in command:
            return "Error:", "Missing 'name' field in 'command' object"

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error:", str(e)


def execute_command(command_name, arguments):
    """Execute the command and return the result"""
    memory = get_memory(cfg)

    try:
        if command_name == "google":
            # Check if the Google API key is set and use the official search method
            # If the API key is not set or has only whitespaces, use the unofficial
            # search method
            key = cfg.google_api_key
            if key and key.strip() and key != "your-google-api-key":
                return google_official_search(arguments["input"])
            else:
                return google_search(arguments["input"])
        elif command_name == "scholar_search":
            return scholar_search(arguments["input"])
        elif command_name == "arxiv_search":
            return arxiv_search(arguments["input"], arguments["sort_by"])
        elif command_name == "eprint_search":
            return eprint_search(arguments["input"], arguments["year"], arguments["area"])
        elif command_name == "download_paper":
            return download_paper(arguments["url"], arguments["file"])
        elif command_name == "read_pdf_write_summary":
            return read_pdf(arguments["file"], arguments["summary_file"])
        elif command_name == "memory_add":
            return memory.add(arguments["string"])
        elif command_name == "start_agent":
            return start_agent(
                arguments["name"], arguments["task"], arguments["prompt"]
            )
        elif command_name == "message_agent":
            return message_agent(arguments["key"], arguments["message"])
        elif command_name == "list_agents":
            return list_agents()
        elif command_name == "delete_agent":
            return delete_agent(arguments["key"])
        elif command_name == "get_text_summary":
            return get_text_summary(arguments["url"], arguments["question"])
        elif command_name == "get_hyperlinks":
            return get_hyperlinks(arguments["url"])
        elif command_name == "read_file":
            return read_file(arguments["file"])
        elif command_name == "write_to_file":
            return write_to_file(arguments["file"], arguments["text"])
        elif command_name == "append_to_file":
            return append_to_file(arguments["file"], arguments["text"])
        elif command_name == "delete_file":
            return delete_file(arguments["file"])
        elif command_name == "search_files":
            return search_files(arguments["directory"])
        elif command_name == "browse_website":
            return browse_website(arguments["url"], arguments["question"])
        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again"
        elif command_name == "evaluate_code":
            return evaluate_code(arguments["code"])
        elif command_name == "improve_code":
            return improve_code(arguments["suggestions"], arguments["code"])
        elif command_name == "write_tests":
            return write_tests(arguments["code"], arguments.get("focus"))
        elif command_name == "execute_python_file":  # Add this command
            return execute_python_file(arguments["file"])
        elif command_name == "execute_shell":
            if cfg.execute_local_commands:
                return execute_shell(arguments["command_line"])
            else:
                return (
                    "You are not allowed to run local shell commands. To execute"
                    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
                    "in your config. Do not attempt to bypass the restriction."
                )
        elif command_name == "generate_image":
            return generate_image(arguments["prompt"])
        elif command_name == "do_nothing":
            return "No action performed."
        elif command_name == "task_complete":
            shutdown()
        else:
            return (
                f"Unknown command '{command_name}'. Please refer to the 'COMMANDS'"
                " list for available commands and only respond in the specified JSON"
                " format."
            )
    # All errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)


def get_datetime():
    """Return the current date and time"""
    return "Current date and time: " + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def google_search(query, num_results=8):
    """Return the results of a google search"""
    search_results = []
    if not query:
        return json.dumps(search_results)

    for j in ddg(query, max_results=num_results):
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def google_official_search(query, num_results=8):
    """Return the results of a google search using the official Google API"""
    import json

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = cfg.google_api_key
        custom_search_engine_id = cfg.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"

    # Return the list of search result URLs
    return search_results_links


def get_text_summary(url, question):
    """Return the results of a google search"""
    text = scrape_text(url)
    summary = summarize_text(url, text, question)
    return """ "Result" : """ + summary


def get_hyperlinks(url):
    """Return the results of a google search"""
    return scrape_links(url)


def shutdown():
    """Shut down the program"""
    print("Shutting down...")
    quit()


def start_agent(name, task, prompt, model=cfg.fast_llm_model):
    """Start an agent with a given name, task, and prompt"""
    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    # Create agent
    if cfg.speak_mode:
        say_text(agent_intro, 1)
    key, ack = agents.create_agent(task, first_message, model)

    if cfg.speak_mode:
        say_text(f"Hello {voice_name}. Your task is as follows. {task}.")

    # Assign task (prompt), get response
    agent_response = agents.message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


def message_agent(key, message):
    """Message an agent with a given key and message"""
    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = agents.message_agent(int(key), message)
    # Check if the key is a valid string
    elif isinstance(key, str):
        agent_response = agents.message_agent(key, message)
    else:
        return "Invalid key, must be an integer or a string."

    # Speak response
    if cfg.speak_mode:
        say_text(agent_response, 1)
    return agent_response


def list_agents():
    """List all agents"""
    return list_agents()


def delete_agent(key):
    """Delete an agent with a given key"""
    result = agents.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."

def scholar_search(query, num_results=8):
    """Return the results of a Google Scholar search"""
    search_results = []
    if not query:
        return json.dumps(search_results)

    search_query = scholarly.search_pubs(query)
    for i, pub in enumerate(search_query):
        if i >= num_results:
            break

        result = {
            "title": pub["bib"].get("title"),
            "author": pub["bib"].get("author"),
            "url": pub["bib"].get("url"),
            "year": pub["bib"].get("year"),
            "abstract": pub["bib"].get("abstract"),
        }

        # # Add the citation in the specified format, if possible
        # try:
        #     citation = scholarly.citation(pub, citation_format)
        #     result["citation"] = citation
        # except Exception as e:
        #     print(f"Error generating citation: {e}")

        search_results.append(result)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def arxiv_search(query: str, sort_by: str = "relevance", num_results: int = 8) -> str:
    search_results = []

    if not query:
        return json.dumps(search_results)

    # if category is not None:
    #     query = f"cat:cs.CL AND {query}"
    
    if sort_by == "relevance":
        sort_criterion = SortCriterion.Relevance
    elif sort_by == "lastUpdatedDate":
        sort_criterion = SortCriterion.LastUpdatedDate
    elif sort_by == "submittedDate":
        sort_criterion = SortCriterion.SubmittedDate
    else:
        sort_criterion = SortCriterion.Relevance

    search = Search(
        query=query,
        max_results=num_results,
        sort_by=sort_criterion,
        sort_order=SortOrder.Descending,
    )

    results = search.results()

    for entry in results:
        result = {
            "title": entry.title,
            "author": ", ".join([author.name for author in entry.authors]),
            "url": str(entry.pdf_url),
            "year": entry.published.year,
            "abstract": entry.summary,
            "arxiv_id": entry.entry_id,
        }
        search_results.append(result)
        download_paper(entry.pdf_url, entry.title+'.pdf')
    return json.dumps(search_results, ensure_ascii=False, indent=4)



def eprint_search(query, year=None, area=None, num_results=8):
    search_results = []
    if not query:
        return json.dumps(search_results)

    base_url = "https://eprint.iacr.org"
    search_url = f"{base_url}/search"  # Updated URL path
    params = {
        "q": query,  # Use 'q' parameter for the search query
        "count": num_results,
    }

    if year:
        params["submittedafter"] = year
        params["submittedbefore"] = year

    if area:
        params["category"] = area

    response = requests.get(search_url, params=params)

    if response.status_code != 200:
        return json.dumps(search_results)

    soup = BeautifulSoup(response.text, "html.parser")
    search_results_elements = soup.select('div.results > div.mb-4')

    if not search_results_elements:
        return json.dumps(search_results)

    for paper in search_results_elements:
        paper_info = {}
        # Extract paper title
        paper_info['title'] = paper.find('strong').text
        # Extract authors
        paper_info['authors'] = paper.find(class_='fst-italic').text
        # Extract abstract
        paper_info['abstract'] = paper.find(class_='mb-0 mt-1 search-abstract').text
        # Extract URL
        paper_info['url'] = f"{base_url}{paper.find(class_='paperlink')['href']}{'.pdf'}"
        search_results.append(paper_info)
        download_paper(paper_info['url'], paper_info['title']+'.pdf')

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def read_pdf(file_path, summary_file_path="Default.txt"):
    """Read text from a PDF file"""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            text = ""

            for page in range(num_pages):
                text += reader.pages[page].extract_text()
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)
            summary = summarize_text(file_path, text, "Summarize the problem statement, the motivation, the approach, and the results of the paper.")
            with open(summary_file_path, "a") as f:
                f.write(summary)
        return """ "Result" : """ + summary
    except Exception as e:
        return f"Error reading PDF file: {e}"


def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks
    
def download_paper(url, file_path):
    """Download a paper from a URL and save it as a PDF file"""
    try:
        response = requests.get(url)

        if response.status_code != 200:
            return f"Error downloading paper: {response.status_code} {response.reason}"

        with open(file_path, "wb") as f:
            f.write(response.content)

        return f"Paper downloaded successfully to {file_path}"
    except Exception as e:
        return f"Error downloading paper: {e}"