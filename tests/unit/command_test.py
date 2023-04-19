import json
from bs4 import BeautifulSoup
import requests

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

    # print("URL requested:", response.url)  # Print the URL being requested
    # with open("eprint_search.txt", "w") as f:
    #     print("HTML response:", response.text, file=f)  # Print the HTML response received

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

    return json.dumps(search_results, ensure_ascii=False, indent=4)

# Test script to call the eprint_search function
def test_eprint_search():
    # Test case 1
    results_1 = eprint_search(query="cryptography", year=2020, area="SECRETKEY", num_results=8)
    print("Test case 1 results:")
    print(results_1)

    # Test case 2
    results_2 = eprint_search(query="quantum computing", num_results=8)
    print("Test case 2 results:")
    print(results_2)

# # Run the test script
# test_eprint_search()

import unittest
import PyPDF2
import os

def read_pdf(file_path):
    """Read text from a PDF file"""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            text = ""

            for page in range(num_pages):
                text += reader.pages[page].extract_text()

        return text
    except Exception as e:
        return f"Error reading PDF file: {e}"

class TestReadPdf(unittest.TestCase):
    def test_read_pdf(self):
        # Create a sample PDF file for testing
        from reportlab.pdfgen import canvas

        test_pdf_file = "test_pdf_file.pdf"
        sample_text = "This is a sample text for testing."

        # Create a PDF file with the desired text
        c = canvas.Canvas(test_pdf_file)
        c.drawString(50, 50, sample_text)  # Add text to the page (coordinates 50, 50)
        c.save()  # Save the PDF file

        # Test the read_pdf function
        extracted_text = read_pdf(test_pdf_file)
        self.assertEqual(extracted_text.strip('\n'), sample_text)

        # Clean up by removing the test PDF file
        os.remove(test_pdf_file)

if __name__ == '__main__':
    unittest.main()
