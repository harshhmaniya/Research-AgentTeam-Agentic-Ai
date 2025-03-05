from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()


research_planner = Agent(
    name="Research Planner",
    role="Breaks research queries into structured subtopics and assigns relevant sources",
    model=Ollama(id="llama3.2"),
    instructions=dedent("""\
        - Decompose research queries into well-structured subtopics covering all key aspects.
        - Ensure logical flow and coverage of historical, current, and future perspectives.
        - Identify and recommend the most credible sources for each subtopic.
        - Prioritize primary research, expert opinions, and authoritative publications.
        - Generate a detailed research roadmap specifying:
          1. Subtopics with clear focus areas.
          2. Recommended sources (websites, papers, reports).
          3. Suggested research methodologies (quantitative, qualitative, case studies).
    """),
    markdown=True
)

research_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    description="An expert researcher conducting deep web searches and verifying sources.",
    instructions=dedent("""
        -Go through the research plan
        -perform relevant web searches based on the planned topics ad resources
        - Prioritize recent and authoritative sources.
        -- Identify key stakeholders and perspectives
    """),
    expected_output=dedent("""
        # Research Summary Report

        ## Topic: {Research Topic}

        ### Key Findings
        - **Finding 1:** {Detailed explanation with supporting data}
        - **Finding 2:** {Detailed explanation with supporting data}
        - **Finding 3:** {Detailed explanation with supporting data}

        ### Source-Based Insights
        #### Source 1: {Source Name / URL}
        - **Summary:** {Concise summary of key points}
        - **Relevant Data:** {Key statistics, dates, or figures}
        - **Notable Quotes:** {Direct citations from experts, if available}

        #### Source 2: {Source Name / URL}
        - **Summary:** {Concise summary of key points}
        - **Relevant Data:** {Key statistics, dates, or figures}
        - **Notable Quotes:** {Direct citations from experts, if available}

        (...repeat for all sources...)

        ### Overall Trends & Patterns
        - **Consensus among sources:** {Common viewpoints and recurring themes}
        - **Diverging Opinions:** {Conflicting perspectives and debates}
        - **Emerging Trends:** {New insights, innovations, or potential shifts in the field}

        ### Citations & References
        - [{Source 1 Name}]({URL})
        - [{Source 2 Name}]({URL})
        - (...list all sources with links...)

        ---
        Research conducted by AI Investigative Journalist
        Compiled on: {current_date} at {current_time}
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)


analysis_agent = Agent(
    model=Ollama(id="llama3.2"),
    description="A data analyst identifying trends, evaluating viewpoints, and spotting inconsistencies.",
    instructions=dedent("""
        - Analyze collected research for patterns, trends, and conflicting viewpoints.
        - Evaluate the credibility of sources and filter out misinformation.
        - Summarize findings with statistical and contextual backing.
    """),
    expected_output=dedent("""A critical analysis report in detail with all the required citations and sources in a proper format""")
)


writing_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="A professional journalist specializing in NYT-style reporting.",
    instructions=dedent("""
        - Write a compelling, well-structured article based on the analysis.
        - Maintain journalistic integrity, objectivity, and balance.
        - Use clear, engaging language and provide necessary background. """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)


editor_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="An editorial assistant verifying accuracy, coherence, and readability.",
    instructions=dedent(""" Check the article generated
        - Verify all facts, statistics, and quotes based on the research analysis report.
        - Ensure smooth narrative flow and logical structure.
        - Check grammar, clarity, and engagement level.
        - Highlight any areas needing further verification or revision.
    """),
    expected_output=dedent("""\
    The same report but with editing instructions in brackets wherever its required
    It will be like each paragraph then editing instructions and suggestion for improvement then next para next set of instruction like that
    -Also maintain the citations"""),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)


research_team = Agent(
    name="A multi-agent journalism team conducting investigative reporting collaboratively.",
    role="Executes a structured research workflow",
    model=Ollama(id="llama3.2"),
    team=[research_planner,research_agent, analysis_agent, writing_agent,editor_agent],
    instructions=dedent("""\
        You are responsible for executing a structured research workflow.
        - Assign tasks to each agent sequentially.
        - Ensure that the output from one agent flows into the next.
        - finally Produce a well-researched, structured final report based on the output from the editor_agent with proper citations.
    """),
    expected_output=dedent("""\
        # {Compelling Headline} 📰

        ## Executive Summary
        {Concise overview of key findings and significance}

        ## Background & Context
        {Historical context and importance}
        {Current landscape overview}

        ## Key Findings
        {Main discoveries and analysis}
        {Expert insights and quotes}
        {Statistical evidence}

        ## Impact Analysis
        {Current implications}
        {Stakeholder perspectives}
        {Industry/societal effects}

        ## Future Outlook
        {Emerging trends}
        {Expert predictions}
        {Potential challenges and opportunities}

        ## Expert Insights
        {Notable quotes and analysis from industry leaders}
        {Contrasting viewpoints}

        ## Sources & Methodology
        {List of primary sources  with the links}
        {Research methodology overview}

        ---
        Compiled by AI Investigative Journalist
        Published: {current_date}
        Last Updated: {current_time}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

if __name__ == "__main__":
    research_team.print_response(
        "Investigate the impact of AI regulation worldwide and its future implications",
        stream=True,
    )