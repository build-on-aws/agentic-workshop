import os
import re
from textwrap import dedent

import yfinance as yf
from crewai.tools import BaseTool
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

from crewai import LLM, Agent, Crew, Process, Task

deepseek_llama = LLM(
    model="sagemaker/jumpstart-dft-deepseek-llm-r1-disti-20250207-153847",
    temperature=0.7,
    max_tokens=4096,
)

bedrock_MICRO_nova = LLM(model="bedrock/us.amazon.nova-micro-v1:0")
bedrock_LITE_nova = LLM(model="bedrock/us.amazon.nova-lite-v1:0")
bedrock_PRO_nova = LLM(model="bedrock/us.amazon.nova-pro-v1:0")


class YahooFinanceTool(BaseTool):
    name: str = "Yahoo Finance Stock Data"
    description: str = (
        "Fetches real-time stock/ETF data, including price, volume, and market cap."
    )

    def _run(self, symbol: str) -> str:
        """Fetches stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            return (
                f"Symbol: {symbol}\n"
                f"Name: {info.get('longName', 'N/A')}\n"
                f"Current Price: {info.get('regularMarketPrice', 'N/A')}\n"
                f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
                f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}\n"
                f"Market Cap: {info.get('marketCap', 'N/A')}\n"
                f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
            )
        except Exception as e:
            return f"Error fetching data: {str(e)}"


#  Instantiate the tool
yahoo_finance_tool = YahooFinanceTool()


class InvestmentReportPDF(FPDF):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def header(self):
        self.image(os.path.join("images", "CatCapital.png"), 10, 8, 33)
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, f"{self.symbol} Investment Analysis Report", align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


class PDFReportTool(BaseTool):
    name: str = "PDF Report Generator"
    description: str = "Generates a professionally formatted PDF investment report"

    def format_text(self, line: str) -> tuple:
        """Determine text format based on line prefix"""
        if line.startswith("# "):
            return ("h1", line[2:], "Arial", "B", 16, True)
        elif line.startswith("## "):
            return ("h2", line[3:], "Arial", "B", 14, True)
        elif line.startswith("### "):
            return ("h3", line[4:], "Arial", "B", 12, False)
        return ("body", line, "Arial", "", 11, False)

    def _run(self, content: str, symbol: str = "STOCK") -> str:
        try:
            pdf = InvestmentReportPDF(symbol)
            pdf.add_page()

            # Process content in a single pass
            content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            for line in lines:
                style, text, font, weight, size, fill = self.format_text(line)

                pdf.set_font(font, weight, size)
                if fill:
                    pdf.set_fill_color(200, 220, 255)
                    pdf.cell(0, 10, text, ln=True, fill=True)
                else:
                    pdf.multi_cell(0, 10, text)
                pdf.ln(5)

            filename = f"{symbol}_investment_report.pdf"
            pdf.output(filename)
            return f"Report successfully generated as {filename}"

        except Exception as e:
            return f"Error generating PDF report: {str(e)}"


# Create the PDF tool instance
pdf_report_tool = PDFReportTool()


data_collector = Agent(
    role="Stock Data Collector",
    goal="Retrieve stock and ETF data from Yahoo Finance.",
    backstory="""
    A data retrieval bot specialized in fetching stock market data.
    """,
    tools=[yahoo_finance_tool],
    llm=bedrock_LITE_nova,
    verbose=True,
)


financial_analyst = Agent(
    role="Financial Analysis Bot",
    goal="Analyze financial data and provide investment insights.",
    backstory="A talented financial analysis bot producing structured insights for a hedge fund.",
    llm=deepseek_llama,
    verbose=True,
)

#  Report Writer Agent (Uses AWS Bedrock NOVA)
report_writer = Agent(
    role="Hedge Fund Report Writer",
    goal="Generate professional reports summarizing financial insights.",
    backstory="An AI-powered financial writer producing hedge fund reports.",
    llm=bedrock_PRO_nova,  # Uses AWS Bedrock NOVA for text generation
    verbose=True,
)

fetch_stock_task = Task(
    description="Fetch stock data for {symbol} using Yahoo Finance Tool.",
    expected_output="Stock price, market cap, and key metrics for {symbol}.",
    agent=data_collector,
    tools=[yahoo_finance_tool],
)

# Task 2: Analyze Stock Data
analyze_stock_task = Task(
    description="""
You are an expert financial analyst tasked with analyzing stock market data and providing insights. Your analysis should be thorough, data-driven, and actionable.

Your task is to analyze this data and provide a comprehensive report. Follow these steps:

1. Review the data carefully, noting any patterns, anomalies, or significant trends.
2. Conduct a detailed analysis, considering factors such as price movements, trading volumes, market capitalization changes, and any other relevant metrics.
3. Identify potential causes for notable changes or patterns in the data.
4. Develop actionable insights and recommendations based on your analysis.

Before providing your final report, think about the following:
- List out key data points and metrics from the stock data.
- Identify potential patterns or anomalies by comparing different time periods or metrics.
- Consider both bullish and bearish arguments based on the data.
It's OK for this section to be quite long.

After your analysis, provide a structured report with the following sections:

1. Overview: A brief summary of the key findings from your analysis.
2. Key Metrics: Important statistical measures and their implications.
3. Trends: Significant patterns or movements observed in the data.
4. Recommendations: Actionable insights for investors or stakeholders based on your analysis.

Use appropriate headers for each section of your report.

Remember to base all your conclusions and recommendations solely on the provided data. Do not introduce external information or assumptions unless explicitly stated in the data.

Please begin your analysis now.
""",
    expected_output="A structured financial analysis including risk assessment and investment potential.",
    agent=financial_analyst,
    context=[fetch_stock_task],
)


generate_report_task = Task(
    description=f"""
You are a professional financial analyst tasked with creating a hedge fund investment report. Your report will be based on thorough financial analysis and will be saved as a PDF document using the PDF Report Generator tool.

Here is the stock symbol for the investment report:
<stock_symbol>
{{symbol}}
</stock_symbol>

Please follow these steps to complete your task:

1. Conduct a comprehensive financial analysis of the stock with the given symbol.
2. Write a detailed investment report based on your analysis.
3. Generate a PDF document of your report.
4. Save the PDF document with a filename that includes the stock symbol.

Before writing the report, conduct your analysis inside <financial_analysis> tags. Include the following steps:

1. List and interpret key financial metrics (e.g., P/E ratio, EPS growth, debt-to-equity ratio).
2. Perform a comparative analysis with industry peers.
3. Outline the company's financial strengths and weaknesses.
4. Consider both bull and bear arguments for the stock.
5. Summarize your key findings and overall financial health assessment.

This structured analysis will ensure a thorough and well-reasoned report.

Your investment report should include the following sections:

1. Executive Summary
2. Company Overview
3. Industry Analysis
4. Financial Performance Analysis
5. Valuation
6. Risk Assessment
7. Investment Recommendation
8. Conclusion

Please ensure that your report is comprehensive, well-structured, and provides valuable insights for hedge fund investment decisions.

After completing your analysis and report, use the PDF Report Generator tool to create the PDF document. The filename of the PDF should follow this format: "Hedge_Fund_Report_<stock_symbol>.pdf", where <stock_symbol> is replaced with the actual stock symbol provided.

Begin your analysis now, followed by the investment report.    
    """,
    expected_output="A professional hedge fund investment report saved as PDF with the stock symbol in the filename.",
    agent=report_writer,
    tools=[pdf_report_tool],
    context=[analyze_stock_task],
)

hedge_fund_crew = Crew(
    agents=[data_collector, financial_analyst, report_writer],
    tasks=[fetch_stock_task, analyze_stock_task, generate_report_task],
    process=Process.sequential,  #  Ensures tasks run in order
)


result = hedge_fund_crew.kickoff(
    inputs={"symbol": "AMZN"}
)  #  Only provide initial input
print(result)
