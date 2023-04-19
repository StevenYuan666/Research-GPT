from autogpt.promptgenerator import PromptGenerator


def get_prompt() -> str:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "~4000 word limit for short term memory. Your short term memory is short, so"
        " immediately save important information to files."
    )
    prompt_generator.add_constraint(
        "If you are unsure how you previously did something or want to recall past"
        " events, thinking about similar events will help you remember."
    )
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint(
        'Exclusively use the commands listed in double quotes e.g. "command name"'
    )

    # Define the command list
    commands = [
        # (
        #     "Google Scholar Search",
        #     "scholar_search",
        #     {"input": "<search>"},
        # ),
        (
            "arXiv Search & Download",
            "arxiv_search",
            {"input": "<search>", "sort_by": "<sort_by>"},
            "Description: Searches arXiv for papers matching the search query and download them to the current working directory. "
        ),
        (
            "ePrint Search & Download",
            "eprint_search",
            {"input": "<search>", "year": "<year>", "area": "<area>"},
            "Description: Searches ePrint for papers matching the search query and download them to the current working directory. "
        ),
        (
            "Read PDF & Write Summary",
            "read_pdf_write_summary",
            {"file": "<file>", "summary_file": "<summary_file>"},
            "Description: Read the text from a PDF file and output a summary of the text to a file.",
        ),
        # (
        #     "Download Paper",
        #     "download_paper",
        #     {"url": "<url>", "file": "<file>"},
        # ),
        # ("Google Search", "google", {"input": "<search>"}, "Description: Search Google. "),
        # (
        #     "Browse Website",
        #     "browse_website",
        #     {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
        #     "Description: Open a website and search for a specific element on the website. ",
        # ),
        (
            "Start GPT Agent",
            "start_agent",
            {"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"},
        ),
        (
            "Message GPT Agent",
            "message_agent",
            {"key": "<key>", "message": "<message>"},
        ),
        ("List GPT Agents", "list_agents", {}),
        ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),
        ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}, "Description: Write text to a file. "),
        ("Read file", "read_file", {"file": "<file>"}, "Description: Read text from a file. "),
        ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}, "Description: Append text to a file. "),
        ("Delete file", "delete_file", {"file": "<file>"}, "Description: Delete a file. "),
        ("Search Files", "search_files", {"directory": "<directory>"}, "Description: Search for files in a directory. "),
        ("Evaluate Code", "evaluate_code", {"code": "<full_code_string>"}, "Description: Evaluate code. "),
        (
            "Get Improved Code",
            "improve_code",
            {"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"},
            "Description: Get improved code."
        ),
        (
            "Write Tests",
            "write_tests",
            {"code": "<full_code_string>", "focus": "<list_of_focus_areas>"},
            "Description: Write tests for code. "
        ),
        ("Execute Python File", "execute_python_file", {"file": "<file>"}, "Description: Execute a Python file. "),
        (
            "Execute Shell Command, non-interactive commands only",
            "execute_shell",
            {"command_line": "<command_line>"},
            "Description: Execute a shell command. "
        ),
        ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}, "Description: Shutdown the agent. "),
        # ("Generate Image", "generate_image", {"prompt": "<prompt>"}),
        ("Do Nothing", "do_nothing", {}, "Description: Do nothing. "),
    ]

    # Add commands to the PromptGenerator object
    for command_label, command_name, args, description in commands:
        prompt_generator.add_command(command_label, command_name, args, description)

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource(
        "Internet access for searches and information gathering."
    )
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks."
    )
    prompt_generator.add_resource("File output.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Continuously review and analyze your actions to ensure you are performing to"
        " the best of your abilities."
    )
    prompt_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior constantly."
    )
    prompt_generator.add_performance_evaluation(
        "Reflect on past decisions and strategies to refine your approach."
    )
    prompt_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
        " the least number of steps."
    )

    # Generate the prompt string
    return prompt_generator.generate_prompt_string()
