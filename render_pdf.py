import sys
from playwright.sync_api import sync_playwright

def generate_pdf(html_path, pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Navigate to the local HTML file
        page.goto(f'file:///{html_path}')
        # We want exactly 1 page of 90cm x 120cm
        page.pdf(
            path=pdf_path,
            width='90cm',
            height='180cm',
            print_background=True,
            page_ranges='1',
            margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'}
        )
        browser.close()
        print(f"Successfully generated {pdf_path}")

if __name__ == "__main__":
    html_file = "d:/fm_paper_works/hcc-vs-hemangioma/poster.html"
    pdf_file = "d:/fm_paper_works/hcc-vs-hemangioma/poster_v2.pdf"
    generate_pdf(html_file, pdf_file)
