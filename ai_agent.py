import os
import textwrap
import re
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import logging
from typing import Tuple, Optional, Dict, List, Union
from datetime import datetime
import sys

# -------------------------
# Setup logging with UTF-8 encoding
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# -------------------------
# Initialize Groq Client
# -------------------------
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logging.info("Groq client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    client = None

# -------------------------
# SQUID GAME INSPIRED COLOR PALETTES
# -------------------------
COLOR_SCHEMES = {
    'squid_game_primary': ['#CE1141', '#00A86E', '#F7C325', '#FF006D', '#7E57C2'],
    'squid_game_vibrant': ['#8C205C', '#BF4B96', '#A6859D', '#0E7373', '#ADD9D1'],
    'neon_electric': ['#FF006D', '#FF77F4', '#4537FF', '#00F5FF', '#39FF14'],
    'sunset_gradient': ['#FF4632', '#F137A6', '#822FFF', '#FF35C4', '#570FDC'],
    'vibrant_pop': ['#2328FF', '#A1FFAA', '#FF295B', '#DBFF33', '#7F31FF'],
    'bold_contrast': ['#9B0030', '#FF6796', '#B6FF17', '#FE5126', '#570FDC'],
    'modern_gradient': ['#667EEA', '#764BA2', '#F093FB', '#4FACFE', '#43E97B'],
    'cyberpunk': ['#FF0080', '#00FFFF', '#FFFF00', '#FF00FF', '#00FF00'],
    'tropical_sunset': ['#FF5994', '#4F4FFF', '#89FFBE', '#FFBAA7', '#FFFD2D']
}

# -------------------------
# Enhanced Matplotlib Styling with Rich Colors
# -------------------------
def setup_plot_style(theme='squid_game_vibrant'):
    """Configure modern matplotlib styling with rich color schemes"""
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'font.size': 12,
        'font.weight': 500,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        'axes.labelweight': 600,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#333333',
        'legend.fancybox': True,
        'legend.shadow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 2.5,
        'axes.edgecolor': '#2A2A2A',
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'grid.linewidth': 1,
        'grid.color': '#CCCCCC',
        'figure.facecolor': 'white',
        'axes.facecolor': '#F8F9FA',
        'text.color': '#1A1A1A',
        'axes.labelcolor': '#1A1A1A',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'lines.linewidth': 3.5,
        'lines.markersize': 10,
        'patch.edgecolor': '#FFFFFF',
        'patch.linewidth': 2
    })
    
    # Set default color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_SCHEMES[theme])

def get_gradient_colors(n_colors, scheme='squid_game_vibrant'):
    """Generate gradient colors from a scheme"""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    base_colors = COLOR_SCHEMES.get(scheme, COLOR_SCHEMES['squid_game_vibrant'])
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Create interpolated gradient
    cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n_colors)
    return [cmap(i/n_colors) for i in range(n_colors)]

# -------------------------
# Enhanced Chart Type Detection
# -------------------------
def detect_chart_type(question: str, df: pd.DataFrame) -> str:
    """
    Intelligently detect the most appropriate chart type using advanced pattern matching.
    
    Args:
        question (str): User's question
        df (pd.DataFrame): Dataset
    
    Returns:
        str: Chart type ('line', 'bar', 'pie', 'scatter', 'heatmap', 'none')
    """
    logging.info("Detecting optimal chart type...")
    q = question.lower()
    
    # Heatmap/Correlation patterns
    if any(word in q for word in ["heatmap", "correlation matrix", "correlate", "relationship between multiple"]):
        return "heatmap"
    
    # Time series patterns
    elif any(word in q for word in ["trend", "time", "month", "year", "quarter", "over time", 
                                     "timeline", "progression", "historical", "evolution", "series"]):
        return "line"
    
    # Distribution patterns
    elif any(word in q for word in ["share", "distribution", "percentage", "proportion", 
                                     "ratio", "composition", "breakdown", "split", "pie"]):
        return "pie"
    
    # Comparison patterns
    elif any(word in q for word in ["compare", "comparison", "top", "bottom", "most", "least", 
                                     "highest", "lowest", "by ", "versus", "rank", "best", "worst"]):
        return "bar"
    
    # Scatter/Correlation patterns
    elif any(word in q for word in ["relationship", "correlation", "scatter", "vs", "against", 
                                     "between", "association", "dependency"]):
        return "scatter"
    
    # Fallback based on data structure
    else:
        num_cols = df.select_dtypes("number").columns
        cat_cols = df.select_dtypes("object").columns
        
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            return "bar"
        elif len(num_cols) >= 2:
            return "line"
        else:
            return "none"

# -------------------------
# Enhanced Data Summary
# -------------------------
def get_data_summary(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataframe structure.
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        str: Detailed summary string
    """
    summary = []
    summary.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()
    
    if numeric_cols:
        summary.append(f"Numeric ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}")
        if len(numeric_cols) > 5:
            summary.append(f"   ...and {len(numeric_cols) - 5} more")
    
    if categorical_cols:
        summary.append(f"Categorical ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}")
        if len(categorical_cols) > 5:
            summary.append(f"   ...and {len(categorical_cols) - 5} more")
    
    if date_cols:
        summary.append(f"Date ({len(date_cols)}): {', '.join(date_cols[:3])}")
    
    # Data quality metrics
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    summary.append(f"Missing data: {missing_pct:.1f}%")
    
    return "\n".join(summary)

# -------------------------
# Advanced Code Cleaner
# -------------------------
def clean_ai_code(code: str) -> str:
    """
    Clean AI-generated code with enhanced logic to remove markdown and explanatory text.
    
    Args:
        code (str): Raw AI output
    
    Returns:
        str: Cleaned Python code
    """
    backtick = chr(96)
    triple_backtick = backtick * 3
    
    code = code.replace(triple_backtick + 'python', '')
    code = code.replace(triple_backtick + 'py', '')
    code = code.replace(triple_backtick, '')
    code = code.strip()
    
    # Remove common AI explanatory phrases at the start
    explanatory_patterns = [
        r"^Here'?s?\s+(?:the|a)\s+(?:code|solution|answer).*?:\s*",
        r"^This\s+(?:code|script|program).*?:\s*",
        r"^Let'?s?\s+.*?:\s*"
    ]
    
    for pattern in explanatory_patterns:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE | re.MULTILINE)
    
    # Split into lines
    lines = code.split('\n')
    code_lines = []
    code_started = False
    code_ended = False
    
    for line in lines:
        stripped = line.strip()
        
        if backtick in stripped and len(stripped) < 15:
            continue
        
        if not code_started and not stripped:
            continue
        
        if not code_started and any(keyword in line for keyword in ['import', 'df', 'plt', 'pd', 'result', '=']):
            code_started = True
        
        if code_started and not code_ended:
            if stripped and not stripped.startswith('#'):
                prose_indicators = [
                    'this code', 'the chart', 'will generate', 'represents', 
                    'the x-axis', 'the y-axis', 'note that', 'this will',
                    'this creates', 'this generates', 'explanation:'
                ]
                
                if any(indicator in stripped.lower() for indicator in prose_indicators):
                    if len(stripped.split()) > 5:
                        code_ended = True
                        continue
                
                if ('=' not in line and 'import' not in line and 'df' not in line and 
                    'plt' not in line and 'pd' not in line and '(' not in line and '[' not in line):
                    if len(stripped.split()) > 5:
                        code_ended = True
                        continue
        
        if code_started and not code_ended:
            code_lines.append(line)
    
    cleaned_code = '\n'.join(code_lines).strip()
    
    lines = cleaned_code.split('\n')
    while lines and lines[-1].strip().startswith('#'):
        last_comment = lines[-1].strip()
        if len(last_comment) > 80 or any(word in last_comment.lower() for word in ['error', 'note', 'explanation']):
            lines.pop()
        else:
            break
    
    cleaned_code = '\n'.join(lines).strip()
    cleaned_code = cleaned_code.replace('plt.show()', '# Chart will be displayed automatically')
    cleaned_code = cleaned_code.replace(backtick, '')
    
    return cleaned_code.strip()

# -------------------------
# Aggressive Code Cleaning Helper
# -------------------------
def aggressive_code_cleaning(code: str) -> str:
    """Remove all non-code lines aggressively"""
    lines = code.split('\n')
    cleaned_lines = []
    
    backtick = chr(96)
    
    for line in lines:
        stripped = line.strip()
        if backtick in stripped and len(stripped) < 15:
            continue
        if (stripped and 
            (stripped.startswith('#') or 
             any(char in line for char in ['=', '(', ')', '[', ']', '.', ',']) or
             any(keyword in stripped for keyword in ['import', 'df', 'plt', 'pd', 'result', 'if', 'for', 'while']))):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# -------------------------
# Enhanced Main AI Function
# -------------------------
def analyze_data(
    df: pd.DataFrame, 
    question: str, 
    chart_choice: str = "auto",
    color_scheme: str = "squid_game_vibrant",
    return_explanation: bool = False,
    temperature: float = 0.1
) -> Union[Tuple[any, str], Tuple[any, str, str]]:
    """
    Analyze user question with enhanced error handling and logging.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        question (str): User's question
        chart_choice (str): Chart type preference
        color_scheme (str): Color scheme name
        return_explanation (bool): Whether to return explanation
        temperature (float): LLM temperature (0.0-1.0)
    
    Returns:
        tuple: (result, code_output) or (result, code_output, explanation)
    """
    logging.info(f"Processing question: '{question}'")
    
    if client is None:
        error_msg = "Groq client not initialized. Please check your GROQ_API_KEY in .env file."
        logging.error(error_msg)
        return error_msg, "# Error: Groq client not initialized"

    try:
        setup_plot_style(color_scheme)
        
        auto_chart = detect_chart_type(question, df)
        chart_type = auto_chart if chart_choice == "auto" else chart_choice
        logging.info(f"Chart type selected: {chart_type}")

        data_summary = get_data_summary(df)
        sample_data = df.head(5).to_string()

        # Get colors for the scheme
        colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['squid_game_vibrant'])
        colors_str = str(colors)

        chart_instruction = f"Create a {chart_type} chart" if chart_type != 'none' else "Do NOT create any charts, just analyze"
        
        prompt = textwrap.dedent(f"""
        You are an expert data scientist specializing in pandas and matplotlib. Generate ONLY executable Python code.
        
        DATASET INFORMATION:
        {data_summary}
        
        Available Columns: {list(df.columns)}
        
        Sample Data (first 5 rows):
        {sample_data}
        
        USER QUESTION: "{question}"
        
        COLOR PALETTE: {colors_str}
        Use these vibrant colors for stunning visualizations!
        
        CRITICAL REQUIREMENTS:
        1. Output ONLY Python code - NO explanations, NO markdown, NO text descriptions
        2. STOP immediately after the last line of code
        3. Use pandas (pd) and matplotlib.pyplot (plt) - both are already imported
        4. The DataFrame is available as 'df' - do not redefine it
        5. Store your final result in a variable named 'result'
        6. {chart_instruction}
        7. NEVER use plt.show() - it's handled automatically
        8. Handle missing values: df.dropna() or df.fillna()
        9. Handle edge cases: check if columns exist, handle empty results
        10. DO NOT include markdown code fences or backticks
        11. For time-based grouping, convert Period objects to strings: .astype(str)
        
        VISUALIZATION BEST PRACTICES (USE THESE EXACT STYLES):
        - Figure Setup: fig, ax = plt.subplots(figsize=(14, 8))
        - Use the provided COLOR PALETTE: colors = {colors_str}
        - Bold title: ax.set_title('Your Title', fontsize=18, fontweight='bold', color='#8C205C', pad=20)
        - Labels: ax.set_xlabel('Label', fontsize=14, fontweight='600', color='#333333')
        - Background: ax.set_facecolor('#F8F9FA')
        - For bar charts: Add thick edges with edgecolor='white', linewidth=2.5
        - For line charts: Add markers with marker='o', linewidth=3.5, markersize=10
        - For pie charts: Use autopct='%1.1f%%', startangle=90, explode slight values, add shadow=True
        - Grid: ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        - Use plt.tight_layout() at the end
        - Add value labels on bars when appropriate
        - For time series: Convert index to string if using Period: result.index = result.index.astype(str)
        
        REMEMBER: Output ONLY executable Python code, no explanations.
        """)

        logging.info("Sending request to Groq LLM...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python data scientist. Generate only executable code without any explanations or markdown formatting. Always convert Period objects to strings for plotting. Use vibrant, rich colors from the provided palette."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=2500,
            top_p=0.9
        )

        code_output = response.choices[0].message.content.strip()
        logging.info(f"Raw AI response received ({len(code_output)} chars)")
        
        code_output = clean_ai_code(code_output)
        logging.info(f"Cleaned code ({len(code_output)} chars)")
        
        is_valid, syntax_error = debug_code_syntax(code_output)
        if not is_valid:
            logging.error(f"Code validation failed: {syntax_error}")
            code_output = aggressive_code_cleaning(code_output)
            
            is_valid, syntax_error = debug_code_syntax(code_output)
            if not is_valid:
                return None, code_output + f"\n\n# Syntax Error: {syntax_error}"

        explanation = f"AI Analysis: Generated Python code to answer '{question}' using {chart_type} visualization with {color_scheme} theme."

        exec_locals = {"df": df.copy(), "pd": pd, "plt": plt, "result": None}
        
        try:
            logging.info("Executing AI-generated code...")
            exec(code_output, {}, exec_locals)
            result = exec_locals.get("result", None)
            
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    if pd.api.types.is_period_dtype(result[col]):
                        result[col] = result[col].astype(str)
                if pd.api.types.is_period_dtype(result.index):
                    result.index = result.index.astype(str)
                    
            elif isinstance(result, pd.Series):
                if pd.api.types.is_period_dtype(result):
                    result = result.astype(str)
                if pd.api.types.is_period_dtype(result.index):
                    result.index = result.index.astype(str)
            
            logging.info("Code execution successful!")

        except KeyError as e:
            result = None
            col_name = str(e).strip("'\"")
            available = list(df.columns)
            error_msg = f"Column '{col_name}' not found.\n\nAvailable columns: {available}"
            logging.error(f"KeyError: {error_msg}")
            code_output += f"\n\n# {error_msg}"
            
        except TypeError as e:
            result = None
            if "Period" in str(e) or "float() argument" in str(e):
                error_msg = "Data contains Period objects. The code needs to convert them to strings."
                logging.error(f"TypeError: {error_msg}")
                code_output += f"\n\n# {error_msg}\n# Fix: Add .astype(str) to convert Period objects"
            else:
                error_msg = f"Type error: {str(e)}"
                logging.error(f"TypeError: {error_msg}")
                code_output += f"\n\n# {error_msg}\n# Tip: Check data types with df.dtypes"
            
        except ValueError as e:
            result = None
            error_msg = f"Data type mismatch: {str(e)}"
            logging.error(f"ValueError: {error_msg}")
            code_output += f"\n\n# {error_msg}\n# Tip: Check data types with df.dtypes"
            
        except ZeroDivisionError as e:
            result = None
            error_msg = "Division by zero detected. Check for zero values in denominator."
            logging.error(error_msg)
            code_output += f"\n\n# {error_msg}"
            
        except Exception as e:
            result = None
            error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
            logging.error(f"Execution Error: {error_msg}")
            code_output += f"\n\n# {error_msg}\n# Tip: Try rephrasing your question"

        if return_explanation:
            return result, code_output, explanation
        else:
            return result, code_output

    except Exception as e:
        logging.error(f"AI Processing Error: {e}")
        error_code = f"# Error during AI processing: {e}\n# Please try:\n# 1. Rephrasing your question\n# 2. Checking your API key\n# 3. Simplifying the query"
        return f"AI Processing Error: {e}", error_code

# -------------------------
# Quick Analysis Functions
# -------------------------
def quick_summary(df: pd.DataFrame) -> Dict[str, any]:
    """Generate comprehensive statistical summary with emojis"""
    return {
        "📊 Total Rows": f"{len(df):,}",
        "📋 Total Columns": len(df.columns),
        "🔢 Numeric Columns": len(df.select_dtypes(include='number').columns),
        "📝 Text Columns": len(df.select_dtypes(include='object').columns),
        "📅 Date Columns": len(df.select_dtypes(include='datetime').columns),
        "⚠️ Missing Values": f"{int(df.isnull().sum().sum())} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%)",
        "🔄 Duplicate Rows": f"{int(df.duplicated().sum())} ({(df.duplicated().sum() / len(df) * 100):.1f}%)",
        "💾 Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }

def suggest_questions(df: pd.DataFrame) -> List[str]:
    """Generate smart suggested questions based on dataset structure"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()
    
    if numeric_cols:
        suggestions.append(f"📊 What is the average {numeric_cols[0]}?")
        suggestions.append(f"📈 Show me the distribution of {numeric_cols[0]}")
        
        if len(numeric_cols) > 1:
            suggestions.append(f"🔗 What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
            suggestions.append(f"📉 Create a scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
    
    if date_cols and numeric_cols:
        suggestions.append(f"⏰ Show me the trend of {numeric_cols[0]} over time")
        suggestions.append(f"📅 What is the monthly total of {numeric_cols[0]}?")
    
    if categorical_cols and numeric_cols:
        suggestions.append(f"🏆 Compare {numeric_cols[0]} by {categorical_cols[0]}")
        suggestions.append(f"🔝 What are the top 10 {categorical_cols[0]} by {numeric_cols[0]}?")
        suggestions.append(f"📊 Show distribution of {categorical_cols[0]}")
    
    if categorical_cols:
        suggestions.append(f"🎯 How many unique {categorical_cols[0]} are there?")
        suggestions.append(f"📋 Show me the frequency of {categorical_cols[0]}")
    
    return suggestions[:10]

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate dataframe with detailed feedback"""
    if df is None:
        return False, "❌ No dataframe provided"
    
    if df.empty:
        return False, "❌ Dataframe is empty - please upload a file with data"
    
    if len(df.columns) == 0:
        return False, "❌ Dataframe has no columns"
    
    if len(df) < 2:
        return False, "❌ Dataframe must have at least 2 rows for analysis"
    
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        return False, f"⚠️ These columns are completely empty: {null_cols}"
    
    return True, "✅ Dataframe is valid and ready for analysis"

def debug_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Debug Python code syntax with detailed error reporting"""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.lineno:
            lines = code.split('\n')
            if 0 < e.lineno <= len(lines):
                error_msg += f"\nProblematic line: {lines[e.lineno - 1]}"
        return False, error_msg
    except Exception as e:
        return False, f"Compilation error: {str(e)}"

# -------------------------
# Export Helper Functions
# -------------------------
def export_analysis_report(df: pd.DataFrame, question: str, result: any, code: str) -> str:
    """Generate formatted analysis report with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if isinstance(result, pd.DataFrame):
        result_str = result.to_string()
    elif isinstance(result, pd.Series):
        result_str = result.to_string()
    elif isinstance(result, (list, dict)):
        result_str = str(result)
    else:
        result_str = str(result)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = len(df.select_dtypes(include='number').columns)
    categorical_cols = len(df.select_dtypes(include='object').columns)
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1]) * 100)
    
    backtick = chr(96)
    triple_backtick = backtick * 3
    
    report_lines = [
        "# AI Data Analysis Report",
        f"**Generated:** {timestamp}",
        "**Powered by:** Groq Llama 3.1 + Squid Game Colors",
        "",
        "---",
        "",
        "## Question",
        f"{question}",
        "",
        "## Dataset Information",
        f"- **Rows:** {total_rows:,}",
        f"- **Columns:** {total_cols}",
        f"- **Numeric Columns:** {numeric_cols}",
        f"- **Categorical Columns:** {categorical_cols}",
        f"- **Missing Values:** {missing_count:,} ({missing_pct:.2f}%)",
        "",
        "## Analysis Result",
        triple_backtick,
        f"{result_str}",
        triple_backtick,
        "",
        "## Generated Code",
        triple_backtick + "python",
        f"{code}",
        triple_backtick,
        "",
        "## Notes",
        "- Analysis is reproducible with the same dataset",
        "- Code can be executed independently",
        "- Results may vary with different data",
        "",
        "---",
        "*Report generated by AI Data Insights Agent v2.0 Enhanced*"
    ]
    
    return "\n".join(report_lines)

def get_column_insights(df: pd.DataFrame, col_name: str) -> Dict[str, any]:
    """Get detailed insights about a specific column"""
    if col_name not in df.columns:
        return {"error": f"Column '{col_name}' not found"}
    
    col = df[col_name]
    insights = {
        "name": col_name,
        "dtype": str(col.dtype),
        "count": len(col),
        "null_count": int(col.isnull().sum()),
        "null_percentage": f"{(col.isnull().sum() / len(col) * 100):.2f}%"
    }
    
    if pd.api.types.is_numeric_dtype(col):
        insights.update({
            "mean": float(col.mean()),
            "median": float(col.median()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "q25": float(col.quantile(0.25)),
            "q75": float(col.quantile(0.75))
        })
    elif pd.api.types.is_object_dtype(col):
        insights.update({
            "unique_count": int(col.nunique()),
            "top_value": str(col.mode()[0]) if len(col.mode()) > 0 else None,
            "top_frequency": int(col.value_counts().iloc[0]) if len(col.value_counts()) > 0 else 0
        })
    
    return insights

# Initialize plot style on import
setup_plot_style()
logging.info("Matplotlib styling configured with Squid Game inspired colors")
