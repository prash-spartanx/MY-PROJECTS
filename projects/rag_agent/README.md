# AI-Powered Research Assistant with Gemini and LangChain

This project is a Python-based research assistant that leverages the power of the Gemini LLM (Large Language Model) from Google, along with the LangChain framework, to automate and streamline research tasks.  The assistant allows users to input a research topic and receive a comprehensive report, including summaries, keywords, and optionally, a generated PDF.

## Features

*   **Topic-Based Research**: Takes a user-provided topic as input.
*   **Dynamic Tool Selection**: Employs an agent to intelligently choose the appropriate tools (e.g., Wikipedia, Arxiv, Web Search) to gather relevant information.
*   **Automated Information Retrieval**:  Retrieves documents and textual data related to the topic from multiple sources.
*   **Text Chunking and Embedding**: Processes retrieved documents by splitting them into smaller chunks and generating embeddings for efficient similarity searches.
*   **Summary and Keyword Generation**: Uses the Gemini LLM to generate a concise summary of the research topic and extract relevant keywords for further investigation.
*   **PDF Report Generation**:  Creates a well-formatted PDF report containing the generated summary, keywords, and optionally, images related to the topic.  (Note: Image scraping is included but is a potentially fragile feature as it relies on Google Images' HTML structure.)
*   **Text-to-Speech (Optional)**: Generates an audio file of the summary using gTTS, which users can download.

## Technologies Used

*   **Python**: The primary programming language.
*   **LangChain**:  A framework for building applications with LLMs.
*   **Google Gemini Pro**: The Large Language Model used for generating summaries and keywords.
*   **langchain-google-genai**: The library to interact with Google's Gemini API.
*   **ChromaDB**:  A vector database for storing and searching embeddings.
*   **Selenium and ChromeDriverManager**:  For web scraping (specifically, the image scraping component).
*   **Beautiful Soup**:  For parsing HTML (when using WebBaseLoader).
*   **FPDF**: For PDF generation.
*   **gTTS**: For Text-to-Speech conversion.
*   **playsound**: For playing audio files.
*   **python-dotenv**: To load environment variables (for API key).
*   **Requests**: To fetch web content.


## Setup and Installation

1.  **Prerequisites**:
    *   A Google Cloud account and a Gemini API key.  You can obtain a key from the Google AI Studio.
    *   Python 3.7 or higher.
    *   The necessary packages installed.

2.  **Install Dependencies**:
    ```bash
    pip install langchain langchain_google_genai langchain-community chromadb fpdf Pillow gTTS requests playsound selenium webdriver-manager
    ```

3.  **Environment Variables**:
    *   Create a `.env` file in the project's root directory (or set them directly in your environment):
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        ```
        *Note: Be careful not to commit your API key to version control.*

4.  **Place the Optional `sample.pdf` (If Using the PDF Tool)**
    *   If you want to test the PDF tool, place a PDF document named `sample.pdf` in the same directory as `agent.py`.  The tool will read this file.

## Running the Project

1.  **Run the Script**:
    ```bash
    python agent.py
    ```

2.  **Follow the Prompts**: The script will prompt you to enter a research topic.

3.  **View the Results**: The script will:
    *   Display a summary and keywords in the console.
    *   Create a PDF file named `output.pdf` in the same directory.
    *   (If the TTS functionality is available)  Play the summary.
    *   (If the image scraping is available) create images folder.

## Important Considerations

*   **Google Gemini API Quotas**: This project uses the Google Gemini API, which has rate limits. You might encounter `429 ResourceExhausted` errors if you exceed your quota.  Consider the rate limits of the used model and your API plan when using the API.
*   **Image Scraping Fragility**: The image scraping component is based on Selenium and is prone to breakage if the Google Images HTML structure changes. This is a common problem with web scraping, and a more robust solution would use a dedicated image search API (if one is available).
*   **PDF Tool**:  The PDF tool is designed for a *specific* PDF file (`sample.pdf`). It's a placeholder.
*   **Local Ollama Integration**:  While this project *uses* the Gemini API, the code includes the setup for using the local Ollama LLM.  This approach avoids API rate limits, and enhances data privacy.

## Potential Improvements

*   **Implement a more reliable image search mechanism.**
*   **Add memory and conversation history to the agent.**
*   **Expand the toolset to include more data sources.**
*   **Improve error handling and reporting.**
*   **Refactor the agent's structure to use LangChain Expression Language (LCEL) for enhanced flexibility and performance.**
*   **Implement more sophisticated retrieval methods (e.g., hybrid search, re-ranking).**

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgements

*   This project uses libraries and frameworks from LangChain, Google, and various open-source projects.


## Author
Prashant Mahableshwar Naik
