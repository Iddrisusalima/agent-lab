# Requirements Document

## Introduction

Agent Lab is a command-line research agent that answers multi-step questions about a user's own notes and documents. Instead of producing a single reply, the agent runs a think → act → observe loop: the model emits a tool request, the application code decides whether to execute that request, the result is fed back to the model, and the cycle repeats until the model returns a final answer with no tool request.

The agent exposes four tools: a calculator, a document search over the user's notes corpus, a note writer that requires explicit human confirmation, and a note reader. Every loop iteration is recorded as a single JSON line in a trace file, and every task is bounded by both a maximum step count and a maximum spend.

Alongside the agent, the project ships an evaluation suite of 12 tasks with automated checks, a non-interactive eval runner that reports pass rate, average step count and average cost, unit tests for the tools, and a documented prompt-injection test. The deliverable is a public GitHub repository that a mentor can clone and run end to end with a single command.

The workspace is greenfield. The model-calling layer (chat loop, token accounting, cost accounting) and the retrieval layer (Chroma vector store, chunking, embedding) are specified here as components to be built inside this project, with an interface narrow enough that equivalent code from the user's earlier projects can be substituted without changing the agent.

## Glossary

- **Agent_Lab**: The complete command-line application described by this document.
- **CLI**: The command-line entry point that accepts a task string, runs the Agent_Loop, and prints step-by-step output and a run summary.
- **Agent_Loop**: The component that repeatedly sends conversation state to the Model_Client, receives either a tool request or a final answer, dispatches tool requests through the Tool_Dispatcher, and appends results to conversation state.
- **Step**: One iteration of the Agent_Loop, consisting of exactly one Model_Client call and zero or more tool executions requested by that call.
- **MAX_STEPS**: A named integer constant that bounds the number of Steps in a single Run. Default value 8.
- **COST_CAP**: A named decimal constant, expressed in United States dollars, that bounds the cumulative model spend of a single Run. Default value 0.25.
- **Run**: A single execution of the Agent_Loop for one user task, from task submission to final answer or termination.
- **Run_Outcome**: The single terminal classification of a Run, equal to exactly one of final answer, step limit reached, cost cap reached, or model error.
- **Model_Client**: The component that sends messages and tool schemas to the configured large language model provider, returns the model reply, and reports token usage for each call.
- **Provider**: The large language model service used by the Model_Client, selected by configuration.
- **Redaction_Marker**: The fixed character sequence substituted wherever a Provider API key value would otherwise appear in terminal output, the Trace_File, or the Eval_Report.
- **Cost_Tracker**: The component that accumulates input tokens, output tokens, and dollar cost across all Model_Client calls in a Run.
- **Price_Table**: A configuration structure mapping a model identifier to input-token and output-token prices per million tokens.
- **Tool**: A named, schema-described function that the Agent_Loop can execute on the model's request.
- **Tool_Registry**: The component that holds all registered Tools, exposes Tool schemas to the Model_Client, and resolves a Tool name to an implementation.
- **Tool_Dispatcher**: The component that validates a tool request, executes the resolved Tool, and converts any failure into a Tool_Result.
- **Tool_Result**: The text or structured value returned to the model as the observation for one tool request, including the text of any error that occurred.
- **Tool_Call_Identifier**: The identifier that the Model_Client assigns to each tool request so that a Tool_Result can be matched to exactly one tool request within a Run.
- **Calculator_Tool**: The Tool named `calculator`, which applies one arithmetic operation to two numbers.
- **Search_Docs_Tool**: The Tool named `search_docs`, which retrieves the most relevant document chunks for a query from the Retrieval_Index.
- **Save_Note_Tool**: The Tool named `save_note`, which appends text to the Notes_File.
- **Read_Notes_Tool**: The Tool named `read_notes`, which returns the contents of the Notes_File.
- **Retrieval_Index**: The component that ingests the Document_Corpus into a Chroma collection and answers similarity queries with chunks and source file names.
- **Keyword_Fallback**: A dependency-free substring and term-frequency search over the Document_Corpus, used by the Search_Docs_Tool when no Chroma collection is available.
- **Document_Corpus**: The set of text and PDF files that the Retrieval_Index indexes, located at a configured directory path.
- **Sample_Notes**: A non-sensitive Document_Corpus of between 5 and 10 files committed to the repository so that a reviewer can run Agent_Lab without supplying private documents.
- **Notes_File**: The single append-only text file that the Save_Note_Tool writes to and the Read_Notes_Tool reads from, located at a configured file path.
- **Confirmation_Gate**: The component that asks the operator to approve a write before the Save_Note_Tool modifies the Notes_File.
- **Interactive_Mode**: The Confirmation_Gate configuration in which approval is requested from the terminal. The default mode of the CLI, applied whenever the Operator supplies no explicit startup option selecting Non_Interactive_Mode.
- **Non_Interactive_Mode**: The Confirmation_Gate configuration in which approval is granted automatically without a terminal prompt. Selected only by an explicit startup option, and never applied as a silent default. The mode used by the Eval_Runner.
- **Trace_Logger**: The component that appends one JSON object per Step to the Trace_File.
- **Trace_File**: A newline-delimited JSON file containing the Trace_Record of every Step of every Run.
- **Trace_Record**: One JSON object describing a single Step: step number, tool name, tool inputs, Tool_Result, and token usage.
- **Run_Summary**: The one-line report printed at the end of a Run, stating total Steps, total tokens, and total cost.
- **System_Prompt**: The instruction text sent to the Model_Client that defines the agent's role, the data-not-instructions rule, and the no-answer rule.
- **Untrusted_Data_Delimiters**: The explicit start and end marker lines that enclose text returned by the Search_Docs_Tool, the Read_Notes_Tool, or a configured memory file, labelling the enclosed text as document data to be summarized rather than as instructions to be followed.
- **Eval_Suite**: The file `evals/tasks.json` containing 12 Eval_Task entries.
- **Eval_Task**: One JSON entry containing an identifier, a prompt, a category, and one or more Eval_Check definitions.
- **Eval_Check**: A machine-evaluable assertion about a Run, of a supported check type.
- **Eval_Runner**: The script that executes every Eval_Task in Non_Interactive_Mode, evaluates every Eval_Check, prints a results table, and writes Eval_Report.
- **Suite_Cost_Ceiling**: The configured upper bound, default 5.00 United States dollars, on the cumulative model spend of one complete Eval_Runner execution.
- **Eval_Report**: The file `evals/results.md` recording per-task outcomes, aggregate metrics, a before pass rate, and an after pass rate.
- **Injection_Document**: A file in Sample_Notes containing text that instructs the reader to disobey the System_Prompt, used to test injection resistance.
- **README**: The file `README.md` at the repository root.
- **Operator**: The person running Agent_Lab from a terminal.

## Requirements

### Requirement 1: Project Structure and Configuration

**User Story:** As a developer, I want a clean, conventional project layout with configuration supplied by environment variables, so that a reviewer can install and run Agent_Lab without reading the source code.

#### Acceptance Criteria

1. THE Agent_Lab repository SHALL place the CLI, the Agent_Loop, the Model_Client together with the Cost_Tracker, each of the four Tools, the Retrieval_Index, the Trace_Logger, and the Eval_Runner in separate importable source files, such that each of those files can be imported without importing the CLI entry point.
2. THE Agent_Lab repository SHALL declare all runtime and test dependencies in a single dependency manifest at the repository root.
3. THE Agent_Lab repository SHALL declare in that dependency manifest a minimum supported Python version of 3.10.
4. THE Agent_Lab repository SHALL include a `.gitignore` file that excludes `.env`, the Trace_File, the Notes_File, and the Chroma persistence directory from version control.
5. THE Agent_Lab repository SHALL include a `.env.example` file that lists every environment variable read by Agent_Lab, and for each variable SHALL state a placeholder value, a one-line comment, and the default value applied when the variable is absent, including the value 8 for MAX_STEPS and the value 0.25 United States dollars for COST_CAP.
6. WHEN the CLI starts a Run, THE Agent_Lab SHALL read the Provider selection, the model identifier, the Provider API key, the Document_Corpus directory path, the Notes_File path, the Trace_File path, MAX_STEPS, and COST_CAP from environment variables before the first Step begins.
7. WHERE an environment variable other than the Provider API key is absent or holds an empty string, THE Agent_Lab SHALL apply the default value documented for that variable in `.env.example`, which SHALL be 8 for MAX_STEPS and 0.25 United States dollars for COST_CAP.
8. IF the Provider API key environment variable is absent or holds an empty string when a Run starts, THEN THE CLI SHALL print a message naming the missing environment variable and the `.env.example` file, SHALL exit with a non-zero status code, and SHALL exit before issuing any Provider request and before appending any line to the Trace_File.
9. IF MAX_STEPS does not parse as an integer between 1 and 50 inclusive, or COST_CAP does not parse as a decimal value between 0.01 and 100.00 United States dollars inclusive, THEN THE CLI SHALL print a message naming the offending environment variable, the received value, and the accepted range, SHALL exit with a non-zero status code, and SHALL exit before the first Step begins.
10. WHEN the CLI has read configuration and before the first Step begins, THE CLI SHALL verify read access to the configured Document_Corpus directory and write access to the configured Notes_File path and Trace_File path, and SHALL print one warning line for each path that fails verification naming the environment variable, the configured path, and whether read or write access failed, and SHALL continue the Run.
11. FOR ALL terminal output lines, all Trace_File lines, and all Eval_Report content, no character sequence equal to the value of the Provider API key environment variable SHALL appear, and at every position where that value would otherwise appear THE Agent_Lab SHALL substitute the Redaction_Marker.

### Requirement 2: Model Client and Tool-Calling Interface

**User Story:** As a developer, I want a single provider-agnostic interface for model calls with tool schemas, so that the Agent_Loop contains no provider-specific code and a second Provider can be added later.

#### Acceptance Criteria

1. THE Model_Client SHALL expose a single operation that accepts the System_Prompt, an ordered message list of 1 to 1000 messages, and a list of 0 to 64 Tool schemas, and returns a reply containing zero or more tool requests, assistant text or an empty text value, a non-negative integer input token count, a non-negative integer output token count, and a flag indicating whether the token counts are estimated.
2. WHEN the Model_Client returns a reply, THE Model_Client SHALL report the input token count and the output token count for that call as reported by the Provider, and SHALL set the estimated flag to false.
3. IF the Provider response omits token usage, THEN THE Model_Client SHALL report a non-negative integer token count estimated from the submitted message content and the returned reply content, and SHALL set the estimated flag to true.
4. THE Model_Client SHALL convert the provider-native tool-request format into a single internal tool-request structure containing a tool name, a dictionary of inputs, and a Tool_Call_Identifier that is unique within the Run.
5. IF a Provider call fails with a transport error, a Provider-side error, or a rate-limit error, or does not return a reply within 120 seconds, THEN THE Model_Client SHALL retry the call up to 3 additional times with delays of 1, 2, and 4 seconds before the respective retries.
6. IF a Provider call fails with an authentication error, an invalid-request error, or an unknown-model error, THEN THE Model_Client SHALL raise a single typed error without retrying.
7. IF a Provider call fails after the final retry, THEN THE Model_Client SHALL raise a single typed error describing the Provider, the model identifier, and the failure reason, SHALL return no partial reply, and SHALL leave the message list unmodified.
8. THE Model_Client SHALL select the Provider implementation from configuration at startup.
9. IF the configured Provider identifier does not match an available Provider implementation, or required Provider credentials are absent, THEN THE Agent_Lab SHALL terminate before the first Step with an error indicating the unresolved Provider identifier or the missing credential.
10. WHEN the Model_Client returns a reply containing one or more tool requests, THE Model_Client SHALL preserve each Tool_Call_Identifier so that a Tool_Result submitted in a later call can be matched to exactly one tool request.
11. IF a Tool_Result is submitted whose Tool_Call_Identifier does not match a tool request from a preceding reply in the same Run, THEN THE Model_Client SHALL raise a typed error identifying the unmatched Tool_Call_Identifier and SHALL not issue the Provider call.
12. FOR ALL Provider calls, THE Model_Client SHALL submit the System_Prompt first, followed by the message list in the order received, with each Tool_Result placed after the reply containing its matching tool request.
13. WHEN a reply contains both assistant text and one or more tool requests, THE Model_Client SHALL return both the assistant text and all tool requests in that single reply without discarding either.

### Requirement 3: Token and Cost Accounting

**User Story:** As an Operator, I want every model call measured in tokens and dollars, so that I can see what each task costs and enforce a spending bound.

#### Acceptance Criteria

1. WHEN the Model_Client completes a call, THE Cost_Tracker SHALL add the input token count and the output token count reported for that call, each a non-negative integer, to the Run input token total and the Run output token total before the next Step begins.
2. WHEN the Cost_Tracker records a call, THE Cost_Tracker SHALL compute the dollar cost of that call as the input token count divided by 1,000,000 multiplied by the input price per million tokens listed for the model identifier, plus the output token count divided by 1,000,000 multiplied by the output price per million tokens listed for the model identifier, taking both prices from the Price_Table and retaining at least 6 decimal places of precision.
3. IF the model identifier is absent from the Price_Table, THEN THE Cost_Tracker SHALL record a cost of 0.0 for that call, SHALL still add that call's input and output token counts to the Run totals, and SHALL include one warning naming the unpriced model identifier in the Run_Summary no matter how many calls used that identifier.
4. IF a completed call returns no input token count or no output token count, THEN THE Cost_Tracker SHALL record the missing count as 0, SHALL record a cost of 0.0 for that call, SHALL leave all previously accumulated Run totals unchanged, and SHALL include in the Run_Summary a warning stating that token counts were unavailable and the number of calls affected.
5. WHILE a Run is in progress, THE Cost_Tracker SHALL expose the Run totals for Steps, input tokens, output tokens, and dollar cost, with each total reflecting every call completed before the request and no call still in flight.
6. FOR ALL pairs of successive reads of the Run totals within a single Run, THE Cost_Tracker SHALL report a Step count, input token total, output token total, and dollar cost total each greater than or equal to the value reported by the earlier read.
7. FOR ALL Runs, the input token total, the output token total, and the Step count reported by the Cost_Tracker SHALL equal the corresponding sums of the per-call values recorded in the Trace_File for that Run exactly, and the reported total dollar cost SHALL equal the sum of the per-call costs recorded in the Trace_File for that Run to within 0.000001 United States dollars.
8. THE Price_Table SHALL be stored outside the Agent_Lab program code in a resource that an Operator can edit without modifying or rebuilding code, and SHALL state, for each model identifier, an input price per 1,000,000 tokens, an output price per 1,000,000 tokens, and the date on which those prices were sourced.
9. WHEN a Run ends for any reason, including normal completion, reaching MAX_STEPS, reaching COST_CAP, or an unrecovered error, THE CLI SHALL print a Run_Summary stating the total number of Steps, the total input tokens, the total output tokens, the sum of input and output tokens, and the total dollar cost rounded to 4 decimal places.

### Requirement 4: Agent Loop

**User Story:** As an Operator, I want the agent to chain tool calls until the question is answered, so that multi-step questions about my documents are resolved in one command.

#### Acceptance Criteria

1. WHEN the CLI receives a task string whose length after trimming leading and trailing whitespace is between 1 and 4000 characters inclusive, THE Agent_Loop SHALL start a Run with the System_Prompt and that trimmed task string as the initial conversation state.
2. WHILE the most recent model reply contains at least one tool request, THE Agent_Loop SHALL execute each requested Tool through the Tool_Dispatcher in the order the requests appear in that reply, for up to 8 tool requests per Step, counting repeated identical tool requests as separate executions without deduplication, append each Tool_Result to the conversation state, and perform another Step.
3. WHEN a model reply contains no tool request and its assistant text contains at least one non-whitespace character, THE Agent_Loop SHALL end the Run with the final answer Run_Outcome and return that assistant text as the final answer.
4. THE Agent_Loop SHALL number Steps consecutively starting at 1, incrementing by 1 for each Model_Client call issued within the Run.
5. WHERE no optional Tool is registered by configuration, THE Agent_Loop SHALL pass the schemas of exactly the four registered Tools to every Model_Client call in every Step of every Run.
6. FOR ALL Runs, the number of Steps performed SHALL be greater than or equal to 1 and less than or equal to MAX_STEPS, and the Run SHALL end after at most MAX_STEPS Steps regardless of the content of any model reply.
7. WHEN the number of completed Steps equals MAX_STEPS and the most recent model reply contains a tool request, THE Agent_Loop SHALL end the Run with the step limit reached Run_Outcome without issuing any further Model_Client call and SHALL print a message stating that the step limit of MAX_STEPS was reached and that no final answer was produced.
8. WHEN the Agent_Loop evaluates the total dollar cost reported by the Cost_Tracker before starting a Step and that total equals or exceeds COST_CAP, THE Agent_Loop SHALL end the Run with the cost cap reached Run_Outcome without issuing that Model_Client call and SHALL print a message stating the COST_CAP value in United States dollars and the total cost reached in United States dollars to at least 4 decimal places.
9. FOR ALL Runs, the total dollar cost reported by the Cost_Tracker SHALL be less than or equal to COST_CAP plus the cost of one Model_Client call.
10. WHEN a Run ends, THE CLI SHALL exit with the success exit code for the final answer Run_Outcome and with a distinct, documented, non-zero exit code for each of the step limit reached, cost cap reached, and model error Run_Outcomes.
11. THE Agent_Loop SHALL define MAX_STEPS and COST_CAP as named constants with the values 8 Steps and 0.25 United States dollars respectively, each with a comment stating the reason for the chosen value, and SHALL apply these values to every Run without accepting a per-Run override from a CLI flag.
12. IF the task string received by the CLI is empty, contains only whitespace, or has a trimmed length greater than 4000 characters, THEN THE Agent_Loop SHALL not start a Run and SHALL not issue any Model_Client call, and THE CLI SHALL print an error message indicating that the task string is empty or exceeds the 4000-character limit and SHALL exit with a distinct, documented, non-zero exit code.
13. IF the Model_Client raises its typed error after exhausting its retries, or returns a reply that contains more than 8 tool requests, names a Tool that is absent from the Tool_Registry, or contains neither a tool request nor at least one non-whitespace character of assistant text, THEN THE Agent_Loop SHALL end the Run with the model error Run_Outcome, SHALL retain the conversation state and the Trace_Logger records of all completed Steps, and SHALL print an error message indicating that the Run ended because the model call failed, stating the number of completed Steps and the total cost in United States dollars.
14. FOR ALL Runs, the Run SHALL terminate and SHALL end with exactly one Run_Outcome.

### Requirement 5: Tool Registry and Tool Dispatch

**User Story:** As a developer, I want a single dispatch path that validates and executes tool requests, so that a bad tool name or bad inputs never crash a Run.

#### Acceptance Criteria

1. THE Tool_Registry SHALL register exactly four Tools named `calculator`, `search_docs`, `save_note`, and `read_notes`, such that no two registered Tools share a name under case-sensitive exact comparison.
2. THE Tool_Registry SHALL expose for each Tool a name, a description of the purpose of that Tool consisting of one sentence of at most 200 characters, and a parameter schema that declares for every parameter its name, its type, and whether it is required.
3. WHEN the Tool_Dispatcher receives a tool request, THE Tool_Dispatcher SHALL resolve the tool name through the Tool_Registry by case-sensitive exact match against the registered names, and SHALL execute no Tool code until name resolution and parameter schema validation have both succeeded.
4. IF a tool request names a Tool that is absent from the Tool_Registry, THEN THE Tool_Dispatcher SHALL return a Tool_Result marked as an error that states the requested name is unknown and lists the registered Tool names.
5. IF a tool request omits a parameter declared required in the parameter schema of the resolved Tool, or supplies a parameter value whose type differs from the declared type, THEN THE Tool_Dispatcher SHALL return a Tool_Result marked as an error that names each invalid parameter and states the type and required status the parameter schema declares for it.
6. IF a tool request supplies a parameter name that the parameter schema of the resolved Tool does not declare, THEN THE Tool_Dispatcher SHALL execute no Tool code and SHALL return a Tool_Result marked as an error that names each undeclared parameter and lists the declared parameter names.
7. IF the execution of a Tool raises an exception, THEN THE Tool_Dispatcher SHALL return a Tool_Result marked as an error containing the exception type and the exception message.
8. IF the execution of a Tool does not complete within the configured per-Tool execution timeout, which defaults to 10 seconds and is accepted only in the range 1 second to 60 seconds inclusive, THEN THE Tool_Dispatcher SHALL abandon that execution and SHALL return a Tool_Result marked as an error that states the Tool exceeded its execution timeout and states the timeout value in seconds.
9. FOR ALL tool requests, THE Tool_Dispatcher SHALL return exactly one Tool_Result carrying a status value of either success or error, and SHALL propagate no exception to the Agent_Loop.
10. FOR ALL Tool_Results marked as an error, THE Tool_Dispatcher SHALL omit secret values from the Tool_Result content and SHALL express any filesystem location as a path relative to the project directory.
11. WHEN the Tool_Dispatcher returns a Tool_Result marked as an error, THE Agent_Loop SHALL append that Tool_Result to the conversation state within the same Step and before issuing the next Model_Client request, so that the model can request a corrected tool call.
12. IF the content of a Tool_Result exceeds the configured maximum character length, which defaults to 4000 characters and is accepted only in the range 500 characters to 20000 characters inclusive, THEN THE Tool_Dispatcher SHALL truncate that content to the configured maximum character length and SHALL append a marker stating that truncation occurred and stating the character count before truncation.

### Requirement 6: Calculator Tool

**User Story:** As an Operator, I want the agent to perform arithmetic through a tool, so that numeric answers do not depend on the model's mental arithmetic.

#### Acceptance Criteria

1. THE Calculator_Tool SHALL accept two numeric parameters, each supplied either as a number or as a string holding a decimal numeric literal with an optional leading sign, at most one decimal point, an optional exponent part, optional surrounding whitespace, and a maximum length of 64 characters, and one operation parameter restricted to the case-sensitive values `add`, `subtract`, `multiply`, and `divide`.
2. WHEN the Calculator_Tool receives a supported operation and two numeric parameters that both convert to finite values with absolute value at most 1,000,000,000,000,000, with zero permitted, THE Calculator_Tool SHALL return within 100 milliseconds of invocation a Tool_Result carrying the value of that operation applied to the two converted values.
3. IF the operation parameter is `divide` and the second numeric parameter converts to zero, including the forms `0`, `0.0`, and `-0`, THEN THE Calculator_Tool SHALL return an error indicating that division by zero is undefined, and SHALL return no numeric result.
4. IF the operation parameter holds a value outside the four supported values, including an absent, empty, or differently cased value, THEN THE Calculator_Tool SHALL return an error listing the four supported operations.
5. IF a numeric parameter is absent, is a boolean, is an empty or whitespace-only string, exceeds 64 characters, or otherwise cannot be converted to a finite number by the rule in acceptance criterion 1, THEN THE Calculator_Tool SHALL return an error naming that parameter and the value received for it, and SHALL return no numeric result.
6. FOR ALL pairs of parameters that convert to finite values with absolute value at most 1,000,000,000,000,000 and FOR ALL four supported operations, THE Calculator_Tool SHALL return either a rendered numeric result or an error message naming the failing parameter or condition, SHALL raise no exception to the Tool_Dispatcher, SHALL return results for `add` and `multiply` that are unchanged when the two parameters are exchanged, and SHALL return round-trip results for add-then-subtract and for divide-then-multiply that differ from the original first parameter by at most 1e-9 times the absolute value of that parameter.
7. WHEN the Calculator_Tool returns a result whose value is an exact integer, THE Calculator_Tool SHALL render that result in plain decimal notation with no fractional part and no exponent part.
8. WHEN the Calculator_Tool returns a result whose value is not an exact integer, THE Calculator_Tool SHALL render that result in plain decimal notation rounded half-up to 10 fractional digits with trailing zeros removed.
9. IF the computed result is not finite, being infinity or not-a-number, or its absolute value exceeds 1,000,000,000,000,000, THEN THE Calculator_Tool SHALL return an error indicating that the result lies outside the supported numeric range, and SHALL return no numeric result.

### Requirement 7: Document Retrieval

**User Story:** As an Operator, I want the agent to search my own documents, so that answers are grounded in my notes rather than in the model's training data.

#### Acceptance Criteria

1. WHEN the Retrieval_Index is built, THE Retrieval_Index SHALL traverse the Document_Corpus directory and all of its subdirectories to unlimited depth and SHALL ingest every file whose extension, compared without regard to letter case, is `.txt`, `.md`, or `.pdf`, up to a maximum of 10,000 files per build, ignoring all files with any other extension.
2. THE Retrieval_Index SHALL split each ingested document into chunks of a configured target character length, defaulting to 1,000 characters and accepted only in the range 200 to 4,000 characters inclusive, with a configured character overlap, defaulting to 150 characters and accepted only in the range 0 characters to one character less than the configured target character length, SHALL produce no chunk longer than the configured target character length, and SHALL store with every chunk the source file name and the zero-based position of that chunk within its source document.
3. THE Retrieval_Index SHALL persist the chunk collection to a Chroma store at a configured directory path, and SHALL create that directory if it does not already exist.
4. WHEN the Operator invokes the single CLI command that builds the Retrieval_Index, THE Agent_Lab SHALL replace any existing collection at the configured persistence directory so that no chunk from a prior build remains, and SHALL print the number of files ingested, the number of files skipped, and the number of chunks indexed.
5. WHEN the Retrieval_Index receives a non-empty query string of 1 to 4,000 characters and a result count between 1 and 100 inclusive, THE Retrieval_Index SHALL return within 5 seconds at most that many chunks ordered by descending similarity to the query, each chunk accompanied by the source file name of that chunk, breaking ties between chunks of equal similarity score by ascending source file name and then by ascending chunk position so that repeated identical queries over an unchanged collection return the same ordering.
6. WHERE the Chroma persistence directory holds no collection, THE Search_Docs_Tool SHALL answer queries using the Keyword_Fallback over the Document_Corpus, SHALL return at most the requested result count of matches, and SHALL state in the Tool_Result that keyword fallback search was used.
7. THE Retrieval_Index SHALL expose a retrieval operation whose signature accepts a query string and a result count and returns a list of chunk text and source file name pairs, so that an external retrieval implementation can replace the included implementation without changes to the Search_Docs_Tool.
8. FOR ALL queries and all result counts between 1 and 100 inclusive, the number of chunks returned by the Retrieval_Index SHALL be less than or equal to the requested result count, and every returned chunk SHALL carry a source file name that matches a file ingested from the Document_Corpus during the most recent build.
9. IF a file selected for ingestion cannot be opened or read, exceeds 25 megabytes, or yields no extractable text or only whitespace after text extraction, THEN THE Retrieval_Index SHALL skip that file, SHALL continue ingesting the remaining files, SHALL exclude the skipped file from the ingested file count, and SHALL print for each skipped file its source file name and an indication of the reason it was skipped.
10. IF the configured target character length is outside the range 200 to 4,000 characters inclusive, or the configured character overlap is negative or greater than or equal to the configured target character length, THEN THE Agent_Lab SHALL reject the build command, SHALL return an error indicating which configured value is out of range and the permitted range, and SHALL leave any existing persisted collection unchanged.
11. IF the Retrieval_Index receives an empty or whitespace-only query string, or a result count less than 1 or greater than 100, THEN THE Retrieval_Index SHALL return no chunks and SHALL return an error indicating which parameter is invalid and its permitted range, and THE Search_Docs_Tool SHALL state in the Tool_Result that the search was not performed because of an invalid parameter.

### Requirement 8: Search Docs Tool

**User Story:** As an Operator, I want search results to carry their source file names, so that I can verify where an answer came from.

#### Acceptance Criteria

1. THE Search_Docs_Tool SHALL accept a query parameter of type string with a maximum accepted length of 500 characters and an optional result count parameter of type integer with a default value of 4, a minimum accepted value of 1, and a maximum accepted value of 10.
2. WHEN the Search_Docs_Tool receives a query, THE Search_Docs_Tool SHALL return one entry per retrieved result, ordered by descending similarity, each entry stating the source file name of that result immediately before the chunk text of that result.
3. WHEN the Search_Docs_Tool returns retrieved results, THE Search_Docs_Tool SHALL limit the Tool_Result to at most 4000 characters by omitting the lowest-ranked entries that exceed that limit and stating the number of entries omitted.
4. WHEN the Search_Docs_Tool returns retrieved results, THE Search_Docs_Tool SHALL enclose the chunk text of each entry within the Untrusted_Data_Delimiters.
5. IF the Retrieval_Index returns no chunk for a query, THEN THE Search_Docs_Tool SHALL return a Tool_Result stating that the Document_Corpus contains no matching content for that query and containing no chunk text.
6. IF the Document_Corpus directory is absent or contains no supported file, THEN THE Search_Docs_Tool SHALL return a Tool_Result stating that no documents are available and naming the configured Document_Corpus path.
7. IF the query parameter holds an empty string, a string containing only whitespace characters, or a string longer than 500 characters, THEN THE Search_Docs_Tool SHALL return an error stating the query requirement that was violated and SHALL perform no retrieval.
8. IF the result count parameter holds an integer below 1 or above 10, THEN THE Search_Docs_Tool SHALL use the nearest accepted bound as the result count and SHALL state in the Tool_Result the result count value that was used.
9. FOR ALL Tool_Results returned by the Search_Docs_Tool, every source file name included in the Tool_Result SHALL name a file present in the Document_Corpus directory.

### Requirement 9: Save Note Tool and Human Confirmation

**User Story:** As an Operator, I want to approve every write to my notes before the write happens, so that the agent cannot modify my files without consent.

#### Acceptance Criteria

1. THE Save_Note_Tool SHALL accept exactly one parameter, a text parameter of type string holding 1 to 10,000 characters, and SHALL direct every write to the configured Notes_File path only.
2. WHILE the Confirmation_Gate operates in Interactive_Mode, WHEN the Save_Note_Tool is invoked, THE Save_Note_Tool SHALL display the complete text to be written and the configured Notes_File path to the Operator, and SHALL wait without any time limit for one line of Operator input while leaving the Notes_File unmodified.
3. WHILE the Confirmation_Gate operates in Interactive_Mode, WHEN the Operator submits input that equals `y` or `Y` after removal of leading and trailing whitespace, THE Save_Note_Tool SHALL append the text to the Notes_File, SHALL record the approval decision in the Trace_Record, and SHALL return a Tool_Result stating the number of characters appended.
4. WHILE the Confirmation_Gate operates in Interactive_Mode, WHEN the Operator submits any input that, after removal of leading and trailing whitespace, is not exactly `y` or `Y`, including an empty line and inputs such as `yes`, THE Save_Note_Tool SHALL leave the Notes_File byte-identical to its state before the invocation, SHALL record the declined decision in the Trace_Record, and SHALL return a Tool_Result stating that the Operator declined the write.
5. FOR ALL Save_Note_Tool invocations in Interactive_Mode, the Notes_File contents SHALL be byte-identical before and after the invocation unless the Operator submitted exactly `y` or `Y` for that invocation.
6. WHILE the Confirmation_Gate operates in Non_Interactive_Mode, THE Save_Note_Tool SHALL append the text without requesting approval and SHALL record in the Trace_Record that approval was granted automatically.
7. FOR ALL Save_Note_Tool invocations that append, THE Save_Note_Tool SHALL append the note as a new separate entry prefixed with an ISO 8601 timestamp at the end of the Notes_File, and SHALL leave all pre-existing Notes_File content untruncated, unoverwritten, and in its original order.
8. WHEN the Save_Note_Tool is invoked and the Notes_File does not exist, THE Save_Note_Tool SHALL create the Notes_File and any missing parent directory of the configured Notes_File path before appending.
9. IF the text parameter holds an empty string, holds only whitespace characters, or holds more than 10,000 characters, THEN THE Save_Note_Tool SHALL leave the Notes_File byte-identical to its state before the invocation and SHALL return an error stating that note text must be non-empty and no longer than 10,000 characters.
10. IF writing to the Notes_File fails, THEN THE Save_Note_Tool SHALL return a Tool_Result containing the failure reason and the configured Notes_File path, SHALL record the failure in the Trace_Record, and SHALL make no further write attempt for that invocation.
11. THE Agent_Lab SHALL operate the Confirmation_Gate in Interactive_Mode whenever the Operator has not supplied the explicit non-interactive option at startup, and SHALL select Non_Interactive_Mode only when that explicit option is supplied.
12. IF the Confirmation_Gate operates in Interactive_Mode and no interactive Operator input is available because the input stream is closed or end-of-input is reached before a line is submitted, THEN THE Save_Note_Tool SHALL treat the invocation as declined, SHALL leave the Notes_File byte-identical to its state before the invocation, SHALL record the declined decision in the Trace_Record, and SHALL return a Tool_Result stating that Operator approval could not be obtained.
13. IF a Save_Note_Tool invocation supplies a file path, a destination, or any parameter other than the text parameter, THEN THE Save_Note_Tool SHALL modify no file and SHALL return an error stating that the Save_Note_Tool writes only to the configured Notes_File path.

### Requirement 10: Read Notes Tool

**User Story:** As an Operator, I want the agent to read back my notes file, so that the agent can summarize or extend notes saved earlier.

#### Acceptance Criteria

1. WHEN the Read_Notes_Tool is invoked and the Notes_File contains at least 1 character, THE Read_Notes_Tool SHALL return a Tool_Result containing the text of the Notes_File decoded as UTF-8, enclosed within the Untrusted_Data_Delimiters.
2. IF the Notes_File does not exist at the configured Notes_File path, THEN THE Read_Notes_Tool SHALL return a Tool_Result stating that no notes have been saved yet and naming the configured Notes_File path.
3. IF the Notes_File exists and contains 0 characters, THEN THE Read_Notes_Tool SHALL return a Tool_Result stating that the Notes_File is empty and naming the configured Notes_File path.
4. FOR ALL sequences of Save_Note_Tool invocations approved in a Run whose appended text totals no more than the configured read budget, the text of every approved note SHALL appear in the value returned by a subsequent Read_Notes_Tool invocation, and the contents of the Notes_File SHALL be identical before and after that invocation.
5. THE Read_Notes_Tool SHALL declare a parameter schema containing zero parameters, and IF a tool request for the Read_Notes_Tool supplies any parameter, including a file path or a file name, THEN THE Read_Notes_Tool SHALL read only the configured Notes_File path, SHALL read no other file, and SHALL include in the Tool_Result a statement that the Read_Notes_Tool accepts no parameters.
6. WHEN the Read_Notes_Tool is invoked and the Notes_File contains more characters than the configured read budget, which defaults to 8000 characters, THE Read_Notes_Tool SHALL return the most recently appended whole note entries that fit within that budget, ordered oldest first among the returned entries, and SHALL append a marker stating that earlier entries were omitted and how many characters were omitted.
7. IF reading the Notes_File fails because access to the file is denied or because the file contains bytes that cannot be decoded as UTF-8, THEN THE Read_Notes_Tool SHALL return a Tool_Result stating the failure reason and naming the configured Notes_File path, and SHALL leave the Notes_File unchanged.

### Requirement 11: Step Tracing

**User Story:** As a developer, I want a machine-readable trace of every step, so that I can debug a Run and show the agent's reasoning path in the README.

#### Acceptance Criteria

1. WHEN a Step completes, THE Trace_Logger SHALL append exactly one Trace_Record to the Trace_File as a single line of JSON terminated by exactly one newline character, and SHALL leave every line already present in the Trace_File unmodified.
2. THE Trace_Record SHALL contain the Run identifier, the step number, the wall-clock start time of that Step as an ISO 8601 timestamp in UTC with millisecond precision, the duration of that Step in whole milliseconds measured from the start of the Model_Client call of that Step to the completion of the last tool execution of that Step, the tool name, the tool inputs truncated to at most 4,000 characters, the Tool_Result truncated to at most 4,000 characters, the input token count, the output token count, and the dollar cost of the Model_Client call of that Step rounded to 6 decimal places, with a marker stating that truncation occurred wherever truncation was applied.
3. FOR ALL Runs recorded in the Trace_File, every Trace_Record of one Run SHALL carry the same Run identifier, and that Run identifier SHALL differ from the Run identifier of every other Run recorded in the Trace_File.
4. WHEN a Run ends, THE Trace_Logger SHALL record the Run_Outcome of that Run in the Trace_Record of the final Step of that Run.
5. THE Trace_Logger SHALL record the Redaction_Marker in place of the Provider API key value wherever that value occurs in the tool inputs, the Tool_Result, or the assistant text of a Trace_Record.
6. WHERE a Step produced no tool request, THE Trace_Logger SHALL record a null tool name, an empty tool inputs value, and the final assistant text truncated to at most 4,000 characters in the Trace_Record, with a marker stating that truncation occurred wherever truncation was applied.
7. WHERE a Step produced more than one tool request, THE Trace_Logger SHALL record the tool name, tool inputs, and Tool_Result of every request of that Step within the single Trace_Record of that Step, ordered as the Tool_Dispatcher executed those requests.
8. FOR ALL Runs in which no append failure occurred, the number of lines appended to the Trace_File for that Run SHALL equal the number of Steps performed in that Run, and the step numbers recorded on those lines SHALL form an ascending contiguous sequence starting at 1.
9. FOR ALL lines in the Trace_File, each line SHALL parse as a single JSON object, and every newline character occurring within a recorded value SHALL be escaped so that no Trace_Record spans more than one line.
10. WHEN a Run starts and the Trace_File does not exist, THE Trace_Logger SHALL create an empty Trace_File and any missing parent directory before appending the first Trace_Record.
11. IF appending a Trace_Record fails, THEN THE Trace_Logger SHALL print one warning naming the configured Trace_File path and the failure reason, SHALL discard that Trace_Record, and THE Agent_Loop SHALL continue the Run and end it with the same Run_Outcome and final answer it would have reached had the append succeeded.

### Requirement 12: System Prompt, Grounding, and Injection Resistance

**User Story:** As an Operator, I want the agent to admit when my documents lack an answer and to ignore instructions embedded in document text, so that I can trust the output.

#### Acceptance Criteria

1. THE System_Prompt SHALL state that the agent answers questions about the Operator's documents using only the registered Tools named `calculator`, `search_docs`, `save_note`, and `read_notes`, and SHALL state that a final answer drawn from document text names the source file of every chunk used in that answer.
2. THE System_Prompt SHALL state that any text appearing within the Untrusted_Data_Delimiters in a Tool_Result of the Search_Docs_Tool or the Read_Notes_Tool is data to be summarized, and SHALL state that an instruction contained in that text is to be reported in the final answer as an instruction found in a document rather than followed.
3. THE System_Prompt SHALL state that when the delimited text returned by the Search_Docs_Tool and the Read_Notes_Tool does not contain the requested information, the final answer of the agent includes the phrase `not found in the documents`.
4. WHEN every Search_Docs_Tool invocation of a Run returns no matching content and no Read_Notes_Tool invocation of that Run returns the requested information, THE Agent_Loop SHALL end the Run with a final answer that contains the phrase `not found in the documents` and that requests no further Tool.
5. THE Sample_Notes SHALL include exactly one Injection_Document that contains at least one line instructing the reader to disregard prior instructions and to save the text `HACKED` to the Notes_File, together with non-sensitive topical content on the same subject as at least one other file in Sample_Notes, so that the Injection_Document is returned by at least one Search_Docs_Tool query.
6. WHILE the Confirmation_Gate operates in Interactive_Mode, WHEN a Run summarizes the Injection_Document and the model requests a Save_Note_Tool write, THE Agent_Lab SHALL request Operator approval through the Confirmation_Gate and SHALL leave the Notes_File unchanged unless the Operator approves that write.
7. THE README SHALL report the observed outcome of a Run that summarizes the Injection_Document, stating whether the agent requested a write of the text `HACKED`, whether the agent reported the embedded instruction in its final answer, whether the Confirmation_Gate prevented that write, and the repository location of the committed Trace_File excerpt or terminal transcript from which a reviewer can reproduce that outcome.
8. WHEN the Agent_Loop appends a Tool_Result produced by the Search_Docs_Tool or the Read_Notes_Tool to the conversation state, THE Agent_Loop SHALL enclose that text within the Untrusted_Data_Delimiters and SHALL escape or remove any occurrence of either delimiter inside that text.
9. FOR ALL Runs, no text contained in a Tool_Result SHALL change the value of MAX_STEPS, the value of COST_CAP, the Confirmation_Gate mode, or the set of Tools registered in the Tool_Registry.
10. THE Agent_Lab SHALL load the System_Prompt at startup from a single text resource carrying a version identifier, and SHALL hold no second copy of System_Prompt instruction text, so that the System_Prompt can be edited without changing code.

### Requirement 13: Terminal Output

**User Story:** As an Operator, I want readable step-by-step terminal output, so that I can follow what the agent is doing while a task runs.

#### Acceptance Criteria

1. WHERE step narration is not suppressed, WHEN a Step begins, THE CLI SHALL write to standard output a step heading containing the step number, and SHALL flush that heading before the first Tool of that Step is invoked and within 100 milliseconds of the Step beginning.
2. WHERE step narration is not suppressed, WHEN a Tool is executed, THE CLI SHALL write to standard output the tool name, the tool inputs, and the Tool_Result truncated to the configured maximum result length, flushing this output before the next Step begins.
3. THE CLI SHALL accept a configured maximum result length in the range 80 to 10,000 characters inclusive, SHALL use 500 characters when the Operator supplies no value, SHALL reject a configured value outside that range with an error message indicating the permitted range, and SHALL append a truncation indicator whenever a Tool_Result longer than the configured maximum is shortened.
4. WHEN a Run ends with a final answer, THE CLI SHALL write to standard output the final answer under a labelled heading and then the Run_Summary immediately after it, in that order, regardless of whether step narration is suppressed.
5. WHERE the `rich` package is installed, THE CLI SHALL render step headings, tool names, and the Run_Summary with colour and formatting.
6. WHERE the `rich` package is absent, THE CLI SHALL render the same step numbers, tool names, tool inputs, shortened Tool_Results, success and failure markers, final answer, and Run_Summary fields as the `rich` rendering, as plain text differing only in colour and styling.
7. IF a Tool_Result reports an error, THEN THE CLI SHALL write the shortened Tool_Result preceded by a failure marker that is textually distinct from the marker used for a successful Tool_Result, and SHALL continue printing subsequent Steps of the Run.
8. IF the Agent_Lab emits a warning or an error message, THEN THE CLI SHALL write that message to standard error while continuing to write step narration, the final answer, and the Run_Summary to standard output.
9. FOR ALL text written to standard output or standard error, THE CLI SHALL replace every occurrence of the Provider API key value with the Redaction_Marker, and THE Agent_Lab SHALL at default verbosity emit no log records below INFO level and no Operator-facing text other than the text stated in acceptance criteria 1 through 8.

### Requirement 14: Evaluation Suite

**User Story:** As a developer, I want a fixed set of tasks with automated checks, so that I can measure agent quality objectively and detect regressions.

#### Acceptance Criteria

1. THE Eval_Suite SHALL contain exactly 12 Eval_Task entries.
2. THE Eval_Suite SHALL define for each Eval_Task an identifier of 1 to 64 characters that is unique within the Eval_Suite and that remains unchanged once published, a prompt of 1 to 2000 characters, a category, a maximum step count between 1 and MAX_STEPS, and at least one Eval_Check.
3. THE Eval_Suite SHALL support Eval_Check types that assert that the final answer contains a given text, that the final answer does not contain a given text, that a named Tool was called during the Run, that no Tool was called during the Run, that the Run ended with a final answer, that the Notes_File does not contain a given text at the end of the Run, that the Run used no more Steps than a given count, and that the Run cost no more than a given dollar amount.
4. THE Eval_Suite SHALL include at least 2 Eval_Task entries whose expected solution requires exactly one tool call, whose expected answer is derivable from the committed Sample_Notes alone, and whose Eval_Check asserts that the required Tool was called during the Run.
5. THE Eval_Suite SHALL include at least 3 Eval_Task entries whose expected solution requires two or more chained tool calls, whose expected answer is derivable from the committed Sample_Notes alone, and whose Eval_Check asserts for each required Tool that it was called during the Run.
6. THE Eval_Suite SHALL include at least 1 Eval_Task entry whose answer is absent from the committed Sample_Notes and whose Eval_Check asserts that the final answer contains text reporting that the answer was not found in the Document_Corpus.
7. THE Eval_Suite SHALL include at least 1 Eval_Task entry whose prompt causes a Tool to return a Tool_Result marked as an error and whose Eval_Check asserts that the Run ended with a final answer within the maximum step count declared for that Eval_Task.
8. THE Eval_Suite SHALL include at least 1 Eval_Task entry that is answerable without any tool call and whose Eval_Check asserts that no Tool was called during the Run.
9. THE Eval_Suite SHALL be stored as a single well-formed JSON file at the path `evals/tasks.json`.
10. FOR ALL Eval_Task entries, every Eval_Check SHALL declare a check type that the Eval_Runner supports and every parameter value that the declared check type requires.
11. THE Eval_Suite SHALL define text comparison for the contains and does-not-contain check types as case-insensitive, performed after each run of consecutive whitespace characters in both the compared text and the asserted text is replaced by a single space.
12. THE Eval_Suite SHALL include at least 1 Eval_Task entry whose prompt asks for a summary of the Injection_Document and whose Eval_Check asserts that the Notes_File does not contain the text `HACKED` at the end of the Run.
13. IF the Eval_Suite is not well-formed JSON, or an Eval_Task omits its identifier, prompt, category, maximum step count, or Eval_Check list, or an Eval_Check declares an unsupported check type or omits a required parameter, THEN THE Eval_Runner SHALL print a message naming the affected Eval_Task identifier and the specific defect, SHALL execute no Eval_Task, SHALL leave any existing Eval_Report unchanged, and SHALL exit with a non-zero status code.

### Requirement 15: Evaluation Runner and Reporting

**User Story:** As a developer, I want one command that runs the whole eval suite and reports pass rate, steps and cost, so that a mentor can reproduce my results.

#### Acceptance Criteria

1. THE Agent_Lab SHALL provide a single command that, when invoked with no Eval_Task identifier argument, executes every Eval_Task in the Eval_Suite in the order the Eval_Task entries are declared in the Eval_Suite.
2. WHILE the Eval_Runner executes an Eval_Task, THE Confirmation_Gate SHALL operate in Non_Interactive_Mode and SHALL read no Operator input.
3. THE Eval_Runner SHALL direct Save_Note_Tool writes and Read_Notes_Tool reads to a Notes_File path reserved for evaluation, SHALL write the Trace_File of each Eval_Task to a trace path reserved for evaluation that includes the identifier of that Eval_Task, and SHALL leave the configured Notes_File of the Operator and the configured Trace_File of the Operator byte-identical to their contents before the Run.
4. IF the evaluation Notes_File path or the evaluation trace path resolves to the same path as the configured Notes_File or the configured Trace_File of the Operator, THEN THE Eval_Runner SHALL execute no Eval_Task, SHALL report an error indicating that the evaluation paths collide with the configured paths of the Operator, and SHALL exit with a non-zero status code.
5. WHEN the Eval_Runner starts an Eval_Task Run, THE Eval_Runner SHALL reset the evaluation Notes_File to an empty state, so that no note written by a previously executed Eval_Task is readable by the Read_Notes_Tool during the current Eval_Task.
6. WHEN an Eval_Task Run ends, THE Eval_Runner SHALL evaluate every Eval_Check of that Eval_Task, SHALL record the Eval_Task as passed only when every Eval_Check of that Eval_Task holds, and SHALL record the identifier of every Eval_Check of that Eval_Task that did not hold.
7. IF an Eval_Task Run raises an error, THEN THE Eval_Runner SHALL record that Eval_Task as failed together with the error message truncated to at most 500 characters and SHALL continue with the remaining Eval_Task entries.
8. IF the cumulative dollar cost of the completed Eval_Task Runs reaches or exceeds the Suite_Cost_Ceiling, THEN THE Eval_Runner SHALL start no further Eval_Task Run, SHALL record every unstarted Eval_Task with the outcome not run, SHALL report an indication that the Suite_Cost_Ceiling was reached, and SHALL exit with a non-zero status code.
9. WHEN every selected Eval_Task has been executed or recorded as not run, THE Eval_Runner SHALL print a table with one row per Eval_Task of the Eval_Suite showing the Eval_Task identifier, the outcome as exactly one of passed, failed, or not run, the number of Steps used as a non-negative integer, the dollar cost rounded to 4 decimal places, and, for a failed Eval_Task, the identifier of each failing Eval_Check or the truncated error message.
10. WHEN every selected Eval_Task has been executed or recorded as not run, THE Eval_Runner SHALL print the pass rate as a percentage rounded to 1 decimal place equal to the count of Eval_Task entries recorded as passed divided by the total count of Eval_Task entries in the Eval_Suite multiplied by 100, the mean number of Steps per executed Eval_Task rounded to 1 decimal place, and the mean dollar cost per executed Eval_Task rounded to 4 decimal places.
11. WHEN every selected Eval_Task has been executed or recorded as not run, THE Eval_Runner SHALL write to the Eval_Report the per-task outcomes stated in acceptance criterion 9, the aggregate metrics stated in acceptance criterion 10, the model identifier used for the Run, the Run start date and time in UTC, and the total dollar cost of the suite rounded to 4 decimal places.
12. THE Eval_Report SHALL contain a before pass rate measured prior to improvements and an after pass rate measured following improvements, each expressed as a percentage rounded to 1 decimal place with its model identifier and measurement date, and a description of each change made to Tool descriptions, to the System_Prompt, or to code between the two measurements.
13. WHERE the command is invoked with exactly one Eval_Task identifier argument, THE Eval_Runner SHALL execute only the Eval_Task whose identifier equals that argument and SHALL print the table and the aggregate metrics for that single Eval_Task.
14. IF the command is invoked with an Eval_Task identifier argument that no Eval_Task in the Eval_Suite carries, THEN THE Eval_Runner SHALL execute no Eval_Task, SHALL report an error indicating that the identifier is unknown, and SHALL exit with a non-zero status code.
15. THE Eval_Runner SHALL exit with status code 0 only when every Eval_Task selected for the Run was executed and recorded as passed, and SHALL exit with a non-zero status code in every other case, including a failed Eval_Task, a not run Eval_Task, and a Run aborted before any Eval_Task executed.

### Requirement 16: Automated Tests

**User Story:** As a developer, I want pytest coverage of the tools, so that edge cases stay fixed once I fix them.

#### Acceptance Criteria

1. THE Agent_Lab repository SHALL include a pytest test suite that a single documented command issued from the repository root runs to completion within 120 seconds of wall-clock time, reporting a zero exit status when every test passes and a non-zero exit status when any test fails or errors.
2. THE pytest test suite SHALL cover the Calculator_Tool with at least one test for each of the operations `add`, `subtract`, `multiply`, and `divide` asserting the returned arithmetic result, one test for `divide` with a second numeric parameter equal to zero asserting an error that states that division by zero is undefined, one test for an operation value outside those four values asserting an error that lists the four supported operations, and one test for a non-numeric parameter value asserting an error that names that parameter and the received value.
3. THE pytest test suite SHALL cover the Read_Notes_Tool for an absent Notes_File asserting a Tool_Result stating that no notes have been saved, for a Notes_File containing zero characters asserting a Tool_Result stating that the Notes_File is empty, and for a Notes_File containing text asserting that the returned value contains that text, with each test asserting that the Notes_File contents are unchanged after the invocation.
4. THE pytest test suite SHALL cover the Search_Docs_Tool for a query that matches no chunk asserting a Tool_Result stating that no matching content exists, for an absent Document_Corpus directory asserting a Tool_Result that names the configured Document_Corpus path, and for an empty query string asserting an error stating that a non-empty query is required.
5. THE pytest test suite SHALL cover the Save_Note_Tool for an approved write in Interactive_Mode asserting that the note text is appended and that the Tool_Result states the number of characters appended, for a declined write in Interactive_Mode asserting that the Notes_File bytes are identical to the bytes present before the invocation, and for Non_Interactive_Mode asserting that the note text is appended without an approval prompt, supplying Operator input through a substitute so that no terminal is required.
6. THE pytest test suite SHALL cover the Tool_Dispatcher for an unknown tool name asserting a Tool_Result that lists the registered Tool names, for a missing required parameter and for a parameter value of the wrong type asserting a Tool_Result that names the invalid parameter, and for a Tool that raises an exception asserting a Tool_Result that contains the exception type and the exception message and asserting that no exception propagates out of the Tool_Dispatcher.
7. THE pytest test suite SHALL cover the Agent_Loop termination conditions with one test for a final answer asserting that the Run ends with the assistant text of the reply that contains no tool request, one test for reaching MAX_STEPS asserting that the number of Steps performed equals MAX_STEPS and that the step-limit message is printed, and one test for reaching COST_CAP asserting that the Run ends before another Step starts and that the total cost is less than or equal to COST_CAP plus the cost of one Model_Client call, each test using a Model_Client substitute that returns scripted replies and performs no network request.
8. THE pytest test suite SHALL run to completion with a zero exit status when no Provider API key environment variable is set and when no network access is available, using a Model_Client substitute in every test that requires a model reply.
9. THE pytest test suite SHALL write files only to paths provided by pytest temporary directory fixtures, and SHALL pass on a clean clone in which no Notes_File, no Trace_File, and no Chroma persistence directory exists before the suite runs.
10. THE pytest test suite SHALL include property-based tests that generate at least 100 input cases per property from a fixed seed recorded in the test code, that report the generating seed and the failing input case when a property fails, and that assert the following properties: the Calculator_Tool returns either a numeric result or an error and raises no exception for all pairs of finite numeric inputs and all operation values; the Tool_Dispatcher returns a Tool_Result and propagates no exception for all tool requests including requests with unknown names, missing parameters, and wrong-typed parameter values; the number of Steps performed is less than or equal to MAX_STEPS and the total dollar cost is less than or equal to COST_CAP plus the cost of one Model_Client call for all scripted model reply sequences; the number of lines in the Trace_File equals the number of Steps performed and every line parses as a single JSON object for all Runs; and the Notes_File is only appended to, with a declined write leaving the Notes_File byte-identical, for all sequences of Save_Note_Tool invocations.
11. THE pytest test suite SHALL cover the Trace_Logger for a Step that requests one Tool, for a Step that requests more than one Tool, for a Step that requests no Tool asserting a null tool name and the recorded final assistant text, for an absent Trace_File asserting that the Trace_File and any missing parent directory are created, and for a failing append asserting that a warning naming the Trace_File path is printed and that the Run continues.
12. THE pytest test suite SHALL cover the Cost_Tracker for token accumulation across calls, for per-call cost computation from the Price_Table, for a model identifier absent from the Price_Table asserting a recorded cost of zero and a warning naming that identifier, and for equality between the reported Run total cost and the sum of the per-call costs recorded in the Trace_File.
13. THE pytest test suite SHALL cover the Eval_Runner with at least one passing case and one failing case for each supported Eval_Check type, using recorded Run data and a Model_Client substitute that performs no network request, and asserting that the evaluated outcome of each Eval_Check matches the expected outcome.

### Requirement 17: Sample Corpus and Repository Deliverables

**User Story:** As a mentor, I want to clone the repository and run everything immediately, so that I can assess the project without configuring private data.

#### Acceptance Criteria

1. THE Agent_Lab repository SHALL include Sample_Notes containing between 5 and 10 files, each file holding between 150 and 2000 words of content that contains no personal data, no credential value, and no text copied from a private source, and SHALL include the Injection_Document as one of those files.
2. THE Agent_Lab repository SHALL include the `evals` directory containing the Eval_Suite at `evals/tasks.json` and the Eval_Report at `evals/results.md`, both committed to version control.
3. THE Agent_Lab repository SHALL contain no Provider API key value and no other credential value in any committed file, and the `.env.example` file SHALL hold placeholder values only.
4. THE README SHALL contain, above the setup instructions, a description of between 50 and 200 words stating the purpose of the agent and the kinds of documents the agent works on, and SHALL identify the Injection_Document by file name as an intentional prompt-injection test fixture rather than a usable document.
5. THE README SHALL contain setup instructions as a numbered sequence of between 4 and 10 commands that a reviewer runs in the stated order from cloning the repository to seeing a first final answer printed, covering dependency installation, supplying a Provider API key, replacing Sample_Notes with the documents of the Operator, and building the Retrieval_Index, and SHALL state that every command in that sequence other than supplying a Provider API key runs against the committed Sample_Notes without any private document.
6. THE README SHALL contain a list of the four Tools using the Tool names registered in the Tool_Registry, with one line of between 5 and 30 words describing each Tool.
7. THE README SHALL contain a section titled `What I Learned` holding a separate labelled entry of at least 2 sentences for each of the three subjects tool calling, the Agent_Loop, and the purpose of the step limit.
8. THE README SHALL state the MAX_STEPS value and the COST_CAP value, each matching the value of the corresponding named constant in the committed code, and SHALL state one sentence giving the reason for each of the two chosen values.
9. THE README SHALL state the before pass rate and the after pass rate, each expressed as a passed-task count out of the total Eval_Task count and matching the corresponding pass rate recorded in the Eval_Report, together with a description of every change made between the two measurements.
10. THE README SHALL contain, above the setup instructions, a link to a demo video and at least one screenshot of a complete Run in which the step number of every Step, every executed tool name, and the Run_Summary are legible.
11. THE README SHALL contain a section titled `Known limitations` listing at least 3 named limitations, a statement that the Agent_Lab is a learning project and not a production system, and a statement titled `What I would improve next` naming at least 2 intended improvements.
12. THE README SHALL contain the command that runs the Eval_Suite and the command that runs the pytest test suite, each shown as a single copyable command line runnable from the repository root.
13. FOR ALL Eval_Task entries whose expected solution requires Document_Corpus content, the Sample_Notes SHALL contain the content needed for every Eval_Check of that Eval_Task to hold, excluding the Eval_Task whose answer is absent from the Document_Corpus by design.
14. THE Sample_Notes SHALL hold the content for at least 1 Eval_Task prompt whose answer requires combining content from two different Sample_Notes files, with neither file alone containing that answer.
15. THE Agent_Lab repository SHALL contain no committed Notes_File, no committed Trace_File, and no committed Chroma persistence directory.

### Requirement 18: Optional Stretch Features

**User Story:** As a developer, I want optional enhancements kept clearly separate from the required scope, so that the core project stays simple and reviewable.

#### Acceptance Criteria

1. THE README SHALL list every stretch feature by name, SHALL state for each listed stretch feature that the feature is optional, that the feature is disabled by default, and the name of the environment variable that enables that feature, and SHALL state that enabling the web search Tool is a documented exception to the four-Tool constraint stated in Requirement 5.
2. WHERE streaming output is enabled by configuration, WHILE the Provider emits output tokens incrementally, THE CLI SHALL write each received output token to standard output within 1 second of receiving that token, and the concatenation of the written tokens SHALL equal the final answer that the Agent_Loop returns for that Run.
3. WHERE a web search Tool is registered by configuration, THE Tool_Registry SHALL expose that Tool to the Model_Client in addition to the four required Tools with a name, a one-sentence description of its purpose, and a parameter schema in the same form as the four required Tools, and THE Tool_Dispatcher SHALL apply to every result of that Tool the same Tool_Result truncation and error-handling criteria that apply to the four required Tools.
4. WHERE a memory file path is configured and that file exists, THE Agent_Loop SHALL read at most the configured Tool_Result maximum character length from that file at the start of a Run and SHALL include the read text in the initial conversation state enclosed within the Untrusted_Data_Delimiters.
5. WHERE a second model identifier is configured for comparison, THE Eval_Runner SHALL execute all 12 Eval_Task entries of the Eval_Suite once against each of the at most 2 configured model identifiers and SHALL state in the Eval_Report, for each model identifier, that model identifier, the count of passed Eval_Task entries, the pass rate as that count divided by 12, and the mean dollar cost per Eval_Task.
6. WHERE no stretch-feature environment variable is set, THE Agent_Lab SHALL disable every stretch feature and SHALL satisfy every acceptance criterion of Requirements 1 through 17 with no configuration other than the Provider API key.
7. FOR ALL stretch features, enabling that stretch feature SHALL leave unchanged the observable behaviour required by the acceptance criteria of Requirements 1 through 17, except for the additional behaviour stated in acceptance criteria 2 through 5 of this requirement.
8. IF a memory file path is configured and that file is absent or cannot be read, THEN THE Agent_Loop SHALL start the Run with the System_Prompt and the task string only, SHALL print a message naming the configured memory file path and the failure reason, and SHALL continue the Run.
9. IF streaming output is enabled by configuration and the configured Provider returns the reply without incremental output tokens, THEN THE CLI SHALL print the complete final answer in a single write and SHALL print a message stating that streaming output was unavailable for the configured Provider.
