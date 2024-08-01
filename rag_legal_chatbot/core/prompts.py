from llama_index.core import PromptTemplate


class ContextPrompt:

    def __call__(self, language: str) -> str:
        if language == "vi":
            return CONTEXT_PROMPT_VI
        elif language == "cs":
            return CONTEXT_PROMPT_CS
        return CONTEXT_PROMPT_EN


CONTEXT_PROMPT_VI = """\
Dưới đây là các tài liệu liên quan cho ngữ cảnh:

{context_str}

Hướng dẫn: Dựa trên các tài liệu trên, cung cấp một câu trả lời chi tiết cho câu hỏi của người dùng dưới đây. \
Trả lời 'không biết' nếu không có trong tài liệu."""

CONTEXT_PROMPT_EN = """\
Here are the relevant documents for the context:

{context_str}

Instruction: Based on the above documents, provide a detailed answer for the user question below. \
Answer 'don't know' if not present in the document."""

CONTEXT_PROMPT_CS = """\
Zde jsou relevantní dokumenty pro kontext:

{context_str}

Instrukce: Na základě výše uvedených dokumentů poskytněte podrobnou odpověď na otázku uživatele níže. \
Odpovězte 'nevím', pokud to není uvedeno v dokumentu."""


class CondensePrompt:

    def __call__(self, language: str) -> str:
        if language == "vi":
            return CONDENSE_PROMPT_VI
        elif language == "cs":
            return CONDENSE_PROMPT_CS
        return CONDENSE_PROMPT_EN


CONDENSE_PROMPT_VI = """\
Cho cuộc trò chuyện sau giữa một người dùng và một trợ lí trí tuệ nhân tạo và một câu hỏi tiếp theo từ người dùng,
đổi lại câu hỏi tiếp theo để là một câu hỏi độc lập.

Lịch sử Trò chuyện:
{chat_history}
Đầu vào Tiếp Theo: {question}
Câu hỏi độc lập:\
"""

CONDENSE_PROMPT_EN = """\
Given the following conversation between a user and an AI assistant and a follow up question from user,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:\
"""

CONDENSE_PROMPT_CS = """\
Vzhledem k následujícímu rozhovoru mezi uživatelem a AI asistentem a následné otázce od uživatele,
přeformulujte následující otázku tak, aby byla samostatnou otázkou.

Historie chatu:
{chat_history}
Následující dotaz: {question}
Samostatná otázka:\
"""


class SystemPrompt:

    def __call__(self, language: str) -> str:
        if language == "vi":
            return SYSTEM_PROMPT_VI
        elif language == "cs":
            return SYSTEM_PROMPT_CS
        return SYSTEM_PROMPT_EN


SYSTEM_PROMPT_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

SYSTEM_PROMPT_VI = """\
Đây là một cuộc trò chuyện giữa người dùng và một trợ lí trí tuệ nhân tạo. \
Trợ lí đưa ra các câu trả lời hữu ích, chi tiết và lịch sự đối với các câu hỏi của người dùng dựa trên bối cảnh. \
Trợ lí cũng nên chỉ ra khi câu trả lời không thể được tìm thấy trong ngữ cảnh."""

SYSTEM_PROMPT_CS = """\
Toto je rozhovor mezi uživatelem a umělou inteligencí. \
Asistent poskytuje užitečné, podrobné a zdvořilé odpovědi na otázky uživatele na základě kontextu. \
Asistent by měl také uvést, když odpověď nelze nalézt v kontextu."""


class QueryGenPrompt:

    def __call__(self, language: str) -> str:
        if language == "vi":
            return QUERY_GEN_PROMPT_VI
        elif language == "cs":
            return QUERY_GEN_PROMPT_CS
        return QUERY_GEN_PROMPT_EN


QUERY_GEN_PROMPT_EN = PromptTemplate(
    "You are a skilled search query generator, dedicated to providing accurate and relevant search queries that are concise, specific, and unambiguous.\n"
    "Generate {num_queries} unique and diverse search queries, one on each line, related to the following input query:\n"
    "### Original Query: {query}\n"
    "### Please provide search queries that are:\n"
    "- Relevant to the original query\n"
    "- Well-defined and specific\n"
    "- Free of ambiguity and vagueness\n"
    "- Useful for retrieving accurate and relevant search results\n"
    "### Generated Queries:\n"
)

QUERY_GEN_PROMPT_VI = PromptTemplate(
    "Bạn là một người tạo truy vấn tìm kiếm tài năng, cam kết cung cấp các truy vấn tìm kiếm chính xác và liên quan, ngắn gọn, cụ thể và không mơ hồ.\n"
    "Tạo ra {num_queries} truy vấn tìm kiếm độc đáo và đa dạng, mỗi truy vấn trên một dòng, liên quan đến truy vấn đầu vào sau đây:\n"
    "### Truy vấn Gốc: {query}\n"
    "### Vui lòng cung cấp các truy vấn tìm kiếm mà:\n"
    "- Liên quan đến truy vấn gốc\n"
    "- Được xác định rõ ràng và cụ thể\n"
    "- Không mơ hồ và không thể hiểu sai\n"
    "- Hữu ích để lấy kết quả tìm kiếm chính xác và liên quan\n"
    "### Các Truy Vấn Được Tạo Ra:\n"
)

QUERY_GEN_PROMPT_CS = PromptTemplate(
    "Jste zručný generátor dotazů k vyhledávání, věnující se poskytování přesných a relevantních dotazů, které jsou stručné, specifické a jednoznačné.\n"
    "Vygenerujte {num_queries} jedinečných a různorodých vyhledávacích dotazů, každý na jednom řádku, souvisejících s následujícím vstupním dotazem:\n"
    "### Původní dotaz: {query}\n"
    "### Uveďte prosím vyhledávací dotazy, které jsou:\n"
    "- Relevantní k původnímu dotazu\n"
    "- Dobře definované a specifické\n"
    "- Bez nejednoznačnosti a vágnosti\n"
    "- Užitečné pro získání přesných a relevantních výsledků vyhledávání\n"
    "### Vygenerované dotazy:\n"
)


class SingleSelectPrompt:

    def __call__(self, language: str) -> str:
        if language == "vi":
            return SINGLE_SELECT_PROMPT_VI
        elif language == "cs":
            return SINGLE_SELECT_PROMPT_CS
        return SINGLE_SELECT_PROMPT_EN


SINGLE_SELECT_PROMPT_EN = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "ONE AND ONLY ONE choice that is most relevant to the query: '{query_str}'\n"
)

SINGLE_SELECT_PROMPT_VI = (
    "Dưới đây là một số lựa chọn được đưa ra, được cung cấp trong một danh sách có số thứ tự "
    "(từ 1 đến {num_choices}), "
    "trong đó mỗi mục trong danh sách tương ứng với một tóm tắt.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Chỉ sử dụng các lựa chọn ở trên và không dùng kiến thức trước đó, hãy chọn "
    "1 và chỉ 1 lựa chọn mà liên quan nhất đến câu truy vấn: '{query_str}'\n"
)

SINGLE_SELECT_PROMPT_CS = (
    "Níže je uvedeno několik možností. Jsou uvedeny v očíslovaném seznamu "
    "(od 1 do {num_choices}), "
    "kde každá položka v seznamu odpovídá shrnutí.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Pouze na základě výše uvedených možností a nikoli na základě předchozích znalostí, vyberte "
    "JEDNU A JEDINOU možnost, která je nejrelevantnější k dotazu: '{query_str}'\n"
)
