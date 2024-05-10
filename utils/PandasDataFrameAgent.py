from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.tools import BaseTool
import ast
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.agents import Tool
import re

output_parser = CommaSeparatedListOutputParser()
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ['SERPAPI_API_KEY'] = '85ce8786996e0fa5568e8c4db622cde5b9e883a9667abf39e3f51e3397a3b8e7'

from langchain_community.utilities import SerpAPIWrapper

import langchain_experimental
from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    MULTI_DF_PREFIX,
    MULTI_DF_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
try:
    pd.set_option("display.max_columns", None)
except ImportError:
    raise ImportError("pandas package not found, please install with `pip install pandas`")

class PandasDataFrameAgent:
    """
    This class provides functionalities to create agents that work with pandas DataFrames. 
    It supports both single and multiple DataFrame inputs, and integrates with LangChain's 
    agent and tool architecture. These agents can be used to generate prompts for language models, 
    perform computations, or execute DataFrame manipulations based on the input provided.
    """

    def __init__(self, llm:BaseLanguageModel, df:pd.DataFrame, prefix:Optional[str], suffix:Optional[str],agent_type:AgentType, callback_manager=None, **kwargs):
        """
        Initializes the Pandas DataFrame Agent with necessary parameters.
        
        :param llm: A BaseLanguageModel instance.
        :param df: A pandas DataFrame or a list of DataFrames.
        :param agent_type: The type of agent to be used (ZeroShot or OpenAIFunctions).
        :param callback_manager: A CallbackManager instance (optional).
        :param kwargs: Additional keyword arguments.
        """
        self.llm = llm
        self.df = self._process_dataframe(df)
        self.prefix = prefix
        self.suffix = suffix
        self.agent_type = agent_type
        self.callback_manager = callback_manager
        self.kwargs = kwargs
        self.validate_dataframe()
        self.agent, self.tools = self.create_agent()
    
    def _process_dataframe(self,df):
        if df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'])
        return df
        
    def validate_dataframe(self):
        """
        Validates if the input is a pandas DataFrame or a list of DataFrames.
        """
        if isinstance(self.df, list):
            for item in self.df:
                if not isinstance(item, pd.DataFrame):
                    raise ValueError(f"Expected pandas object, got {type(self.df)}")
        elif not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(self.df)}")

    def create_agent(self):
        """
        Creates and returns the appropriate agent and tools based on the agent type.
        """
        if self.agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            prompt, tools = self._get_prompt_and_tools(self.df)
            llm_chain = LLMChain(llm=self.llm, prompt=prompt, callback_manager=self.callback_manager)
            tool_names = [tool.name for tool in tools]
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, callback_manager=self.callback_manager, **self.kwargs)
        elif self.agent_type == AgentType.OPENAI_FUNCTIONS:
            prompt, tools = self._get_functions_prompt_and_tools(df = self.df,prefix=self.prefix,suffix=self.suffix)
            agent = OpenAIFunctionsAgent(llm=self.llm, prompt=prompt, tools=tools, callback_manager=self.callback_manager, **self.kwargs)
        else:
            raise ValueError(f"Agent type {self.agent_type} not supported at the moment.")
        return agent, tools

    def _get_multi_prompt(
    self,
    dfs: List[Any],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    ) -> Tuple[BasePromptTemplate, List[BaseTool]]:
        num_dfs = len(dfs)
        if suffix is not None:
            suffix_to_use = suffix
            include_dfs_head = True
        elif include_df_in_prompt:
            suffix_to_use = SUFFIX_WITH_MULTI_DF
            include_dfs_head = True
        else:
            suffix_to_use = SUFFIX_NO_DF
            include_dfs_head = False
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad", "num_dfs"]
            if include_dfs_head:
                input_variables += ["dfs_head"]

        if prefix is None:
            prefix = MULTI_DF_PREFIX

        df_locals = {}
        for i, dataframe in enumerate(dfs):
            df_locals[f"df{i + 1}"] = dataframe
        tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix_to_use,
            input_variables=input_variables,
        )
        partial_prompt = prompt.partial()
        if "dfs_head" in input_variables:
            dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
            partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
        if "num_dfs" in input_variables:
            partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
        return partial_prompt, tools

    def _get_single_prompt(
        self,
        df: Any,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        include_df_in_prompt: Optional[bool] = True,
        number_of_head_rows: int = 5,
        extra_tools: Sequence[BaseTool] = (),
    ) -> Tuple[BasePromptTemplate, List[BaseTool]]:
        if suffix is not None:
            suffix_to_use = suffix
            include_df_head = True
        elif include_df_in_prompt:
            suffix_to_use = SUFFIX_WITH_DF
            include_df_head = True
        else:
            suffix_to_use = SUFFIX_NO_DF
            include_df_head = False

        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
            if include_df_head:
                input_variables += ["df_head"]

        if prefix is None:
            prefix = PREFIX
            
        df_locals = {}
        for i, dataframe in enumerate([df]):
            df_locals[f"df{i + 1}"] = dataframe
        tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)
        
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix_to_use,
            input_variables=input_variables,
        )

        partial_prompt = prompt.partial()
        if "df_head" in input_variables:
            partial_prompt = partial_prompt.partial(
                df_head=str(df.head(number_of_head_rows).to_markdown())
            )
        return partial_prompt, tools


    def _get_prompt_and_tools(
        self,
        df: Any,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        include_df_in_prompt: Optional[bool] = True,
        number_of_head_rows: int = 5,
        extra_tools: Sequence[BaseTool] = (),
    ) -> Tuple[BasePromptTemplate, List[BaseTool]]:
        
        params = {
        "engine": "bing",
        "gl": "us",
        "hl": "en",
            }
        search = SerpAPIWrapper(params=params)
        extra_tools = [
            Tool(
                name="Financial Metric Search",
                func=search.run,
                description="useful for when you need to answer queries with complicated financial metric such as maximum drawdown or sharpe ratio",
            )
        ]
        try:
            import pandas as pd

            pd.set_option("display.max_columns", None)
        except ImportError:
            raise ImportError(
                "pandas package not found, please install with `pip install pandas`"
            )
        if include_df_in_prompt is not None and suffix is not None:
            raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

        if isinstance(df, list):
            for item in df:
                if not isinstance(item, pd.DataFrame):
                    raise ValueError(f"Expected pandas object, got {type(df)}")
            print("no multi-df allowed")
        else:
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
            return self._get_single_prompt(
                df,
                prefix=prefix,
                suffix=suffix,
                input_variables=input_variables,
                include_df_in_prompt=include_df_in_prompt,
                number_of_head_rows=number_of_head_rows,
                extra_tools=extra_tools,
            )

    def _get_functions_single_prompt(
        self,
        df: Any,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        include_df_in_prompt: Optional[bool] = True,
        number_of_head_rows: int = 5,
    ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
        if suffix is not None:
            suffix_to_use = suffix
            if include_df_in_prompt:
                suffix_to_use = suffix_to_use.format(
                    df_head=str(df.head(number_of_head_rows).to_markdown())
                )
        elif include_df_in_prompt:
            suffix_to_use = FUNCTIONS_WITH_DF.format(
                df_head=str(df.head(number_of_head_rows).to_markdown())
            )
        else:
            suffix_to_use = ""

        if prefix is None:
            prefix = PREFIX_FUNCTIONS

        tools = [PythonAstREPLTool(locals={"df": df})]
        system_message = SystemMessage(content=prefix + suffix_to_use)
        prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
        return prompt, tools


    def _get_functions_prompt_and_tools(
        self,
        df: Any,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        include_df_in_prompt: Optional[bool] = True,
        number_of_head_rows: int = 5,
    ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
        try:
            import pandas as pd

            pd.set_option("display.max_columns", None)
        except ImportError:
            raise ImportError(
                "pandas package not found, please install with `pip install pandas`"
            )
        if input_variables is not None:
            raise ValueError("`input_variables` is not supported at the moment.")

        if include_df_in_prompt is not None and suffix is not None:
            raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

        if isinstance(df, list):
            for item in df:
                if not isinstance(item, pd.DataFrame):
                    raise ValueError(f"Expected pandas object, got {type(df)}")
            return self._get_functions_multi_prompt(
                df,
                prefix=prefix,
                suffix=suffix,
                include_df_in_prompt=include_df_in_prompt,
                number_of_head_rows=number_of_head_rows,
            )
        else:
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
            return self._get_functions_single_prompt(
                df,
                prefix=prefix,
                suffix=suffix,
                include_df_in_prompt=include_df_in_prompt,
                number_of_head_rows=number_of_head_rows,
            )

    def get_agent(self, **executor_kwargs):
        """
        Executes the agent with provided tools and optional execution parameters.
        """
        
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            callback_manager=self.callback_manager,
            allowed_tools=self.tools,
            **self.kwargs,
            **executor_kwargs,
        )
        
    def extract_tickers(self,text: str) -> list:
        """
        Extracts stock ticker symbols from a given text string.
        
        :param text: String containing stock ticker symbols.
        :return: A list of extracted stock ticker symbols.
        """
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, text)
            
        return tickers

    def parse_agent_output(self, output: str) -> pd.DataFrame:
        """
        Parses the agent's output string to perform DataFrame operations or extract a subset.
        
        :param output: The output string from the agent.
        :return: A pandas DataFrame or a list of DataFrame indices based on the agent's output.
        """
        try:
            result = ast.literal_eval(output)
            if isinstance(result, list):
                return result
            elif isinstance(result, pd.DataFrame):
                return result
            else:
                raise ValueError("Unsupported output format from agent.")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing agent output: {e}")
            return pd.DataFrame()
        
    def run(self, query: str) -> pd.DataFrame:
        """
        Executes a query using the agent and returns a DataFrame or a list of indices from the DataFrame.
        
        :param query: The query string to be executed by the agent.
        :return: A pandas DataFrame or a list of DataFrame indices based on the agent's output.
        """
        agent = self.get_agent()
        prefix = "This DataFrame is a collection of stock market records, where each row corresponds to a specific stock on a specific date, with no unique key. Multiple rows may have the same date or the same stock ticker symbol, but each row combines these two attributes to report daily trading data (opening, closing, high, low, and adjusted closing prices) and volume for that stock on that day,meaning you should do some groupby operations for any query that requires certain lookback. If you are unsure about the query, you can use extra tools. the extra tools are: Financial Metric Search -> helps to find for finance related words. Strictly follow this rule, respond with a Python list of the tickers as the output that satisfy this query:"
        output = agent.run(prefix+query)  # Assuming this returns a string representation of the desired output
        return self.extract_tickers(output)


# from langchain.chat_models import ChatOpenAI
df = pd.read_csv('utils/stock_database.csv')
agent = PandasDataFrameAgent(
    ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key='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'),
    df = df,
    verbose=True,
    prefix=None,
    suffix=None,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)