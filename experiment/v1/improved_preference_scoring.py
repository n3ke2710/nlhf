"""
Улучшенная эвристика для создания preference pairs для TL;DR задачи
Заменяет простое "короче = лучше" на многофакторную оценку
"""

import re
from typing import Dict, List


def score_summary(summary: str, prompt: str) -> float:
    """
    Эвристическая оценка качества TL;DR summary.
    
    Args:
        summary: Сгенерированное резюме
        prompt: Исходный текст (для контекста)
    
    Returns:
        float: Оценка качества (выше = лучше)
    """
    score = 0.0
    
    # Извлекаем только POST часть из промпта (убираем SUBREDDIT, TITLE и т.д.)
    post_match = re.search(r'POST:\s*(.*?)\s*TL;DR:', prompt, re.DOTALL)
    if post_match:
        post_text = post_match.group(1).strip()
    else:
        post_text = prompt
    
    post_length = len(post_text)
    summary_length = len(summary.strip())
    
    # ===== 1. ДЛИНА: оптимальная компрессия =====
    # Идеал: 10-20% от оригинала (для TL;DR)
    min_ideal = post_length * 0.08
    max_ideal = post_length * 0.25
    
    if min_ideal <= summary_length <= max_ideal:
        score += 10  # В целевом диапазоне
    else:
        # Штраф за слишком короткие/длинные
        if summary_length < min_ideal:
            penalty = (min_ideal - summary_length) / min_ideal
            score -= penalty * 8
        else:
            penalty = (summary_length - max_ideal) / max_ideal
            score -= penalty * 5
    
    # ===== 2. СОДЕРЖАТЕЛЬНОСТЬ: overlap с ключевыми словами =====
    # Удаляем стоп-слова для лучшего сравнения
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it',
                  'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'this', 'that'}
    
    post_words = set(w.lower() for w in post_text.split() if w.lower() not in stop_words)
    summary_words = set(w.lower() for w in summary.split() if w.lower() not in stop_words)
    
    if len(post_words) > 0:
        overlap_ratio = len(post_words & summary_words) / len(post_words)
        score += overlap_ratio * 15  # Важный фактор
    
    # ===== 3. ЧИТАЕМОСТЬ: структура и грамматика =====
    summary_clean = summary.strip()
    
    # Заканчивается правильно
    if summary_clean and summary_clean[-1] in '.!?':
        score += 3
    elif summary_clean and summary_clean[-1] in ',;:':
        score -= 2  # Обрыв
    
    # Нет странных артефактов
    if '\n\n' in summary or summary.count('\n') > 2:
        score -= 3  # Слишком много переносов
    
    if '...' in summary or summary.count('.') > 5:
        score -= 2  # Многоточия или слишком много предложений
    
    # Начинается с заглавной буквы
    if summary_clean and summary_clean[0].isupper():
        score += 1
    
    # ===== 4. РАЗНООБРАЗИЕ СЛОВ: нет повторов =====
    summary_words_list = summary.lower().split()
    if len(summary_words_list) > 0:
        unique_ratio = len(set(summary_words_list)) / len(summary_words_list)
        score += unique_ratio * 5
        
        # Штраф за явные повторы подряд
        for i in range(len(summary_words_list) - 1):
            if summary_words_list[i] == summary_words_list[i + 1]:
                score -= 3
    
    # ===== 5. СПЕЦИФИЧНОСТЬ: не слишком общее =====
    # Плохие шаблонные фразы
    generic_phrases = [
        'need advice', 'what should i do', 'help me', 'not sure what to do',
        'any advice', 'thoughts?', 'opinions?', 'tell me what to do'
    ]
    
    summary_lower = summary.lower()
    for phrase in generic_phrases:
        if phrase in summary_lower:
            score -= 4  # Слишком общее, не резюмирует
    
    # ===== 6. КОГЕРЕНТНОСТЬ: не просто набор слов =====
    # Простая проверка: есть ли глаголы (очень грубая)
    common_verbs = ['is', 'was', 'are', 'were', 'do', 'did', 'have', 'has', 'had',
                    'want', 'need', 'get', 'got', 'make', 'made', 'think', 'know']
    
    has_verb = any(verb in summary_lower.split() for verb in common_verbs)
    if has_verb:
        score += 3
    
    return score


def create_better_preference_pairs(
    prompts: List[str],
    generated_responses: Dict[str, List[str]],
    pairs_per_prompt: int = 10
) -> List[Dict[str, str]]:
    """
    Создает preference pairs используя улучшенную эвристику.
    
    Args:
        prompts: Список промптов
        generated_responses: {policy_id: [responses]} - генерации от разных политик
        pairs_per_prompt: Сколько пар создать для каждого промпта
    
    Returns:
        List[Dict]: Список preference pairs с ключами 'prompt', 'chosen', 'rejected'
    """
    import random
    
    preference_pairs = []
    policy_ids = list(generated_responses.keys())
    
    for idx, prompt in enumerate(prompts):
        for _ in range(pairs_per_prompt):
            # Выбираем две случайные политики
            p1, p2 = random.sample(policy_ids, 2)
            
            r1 = generated_responses[p1][idx]
            r2 = generated_responses[p2][idx]
            
            # Оцениваем оба ответа
            score1 = score_summary(r1, prompt)
            score2 = score_summary(r2, prompt)
            
            # Выбираем лучший
            if score1 > score2:
                chosen, rejected = r1, r2
                score_diff = score1 - score2
            else:
                chosen, rejected = r2, r1
                score_diff = score2 - score1
            
            # Отфильтровываем пары со слишком маленькой разницей
            # (такие пары не дают сильного сигнала для обучения)
            if score_diff < 2.0:
                continue
            
            preference_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    
    return preference_pairs


# ===== ПРИМЕР ИСПОЛЬЗОВАНИЯ В НОУТБУКЕ =====
"""
# Замените код создания preference_pairs на:

from improved_preference_scoring import create_better_preference_pairs

print("Creating preference pairs with improved scoring...")
preference_pairs = create_better_preference_pairs(
    prompts=prompts,
    generated_responses=generated_responses,
    pairs_per_prompt=15  # Увеличено с 5
)

print(f"✅ Created {len(preference_pairs)} preference pairs")
logger.info(f'Created {len(preference_pairs)} preference pairs with improved scoring')

# Далее как обычно - добавляем reference log probs и обучаем
"""


# ===== АЛЬТЕРНАТИВА: Использование готового датасета =====
def load_real_tldr_preferences(max_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Загружает настоящие предпочтения людей из OpenAI TL;DR dataset.
    
    Args:
        max_samples: Максимальное количество примеров (для ограничения размера)
    
    Returns:
        List[Dict]: Preference pairs в формате NLHF
    """
    from datasets import load_dataset # type: ignore
    
    print(f"Loading OpenAI TL;DR preferences dataset (max {max_samples} samples)...")
    
    # Загружаем датасет с человеческими предпочтениями
    dataset = load_dataset(
        "openai/summarize_from_feedback",
        "comparisons",
        split=f"train[:{max_samples}]"
    )
    
    preference_pairs = []
    
    for item in dataset:
        # Структура датасета:
        # - info['post']: исходный Reddit пост
        # - info['title']: заголовок поста
        # - info['subreddit']: сабреддит
        # - summaries: два summary для сравнения
        # - choice: какой summary лучше (0 или 1)
        
        # Формируем промпт в том же формате, что и в коде
        post = item['info']['post']
        title = item['info'].get('title', '')
        subreddit = item['info'].get('subreddit', '')
        
        prompt = f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"
        
        # Извлекаем выбранный и отвергнутый summary
        summaries = item['summaries']
        choice = item['choice']
        
        chosen = summaries[choice]['text'].strip()
        rejected = summaries[1 - choice]['text'].strip()
        
        preference_pairs.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"✅ Loaded {len(preference_pairs)} real preference pairs from humans")
    return preference_pairs


# ===== ИСПОЛЬЗОВАНИЕ НАСТОЯЩЕГО ДАТАСЕТА =====
"""
# В ноутбуке можно заменить на:

from improved_preference_scoring import load_real_tldr_preferences

# Вместо генерации своих пар, используем настоящие
preference_pairs = load_real_tldr_preferences(max_samples=3000)

# Далее ВАЖНО: нужно compute reference log probs для этих пар
# (код для этого уже есть в ноутбуке, просто используйте его)
"""


if __name__ == "__main__":
    # Тест функции
    test_prompt = """SUBREDDIT: r/relationships
TITLE: How do I handle a difficult situation?
POST: My friend is always late to our meetings. We've been friends for 5 years and I really value our friendship, but this is starting to bother me. Last week she was 45 minutes late and didn't even apologize. I don't know if I should say something or just let it go.
TL;DR:"""
    
    test_summary_good = "Friend is chronically late, not sure if I should confront her about it."
    test_summary_bad = "I have a friend and we meet sometimes and there are some issues."
    test_summary_too_long = "My friend is always late to our meetings and we have been friends for a very long time, about 5 years or so, and I really like spending time with her but the lateness is becoming a problem for me."
    
    print("Testing summary scoring:\n")
    print(f"Good summary: {score_summary(test_summary_good, test_prompt):.2f}")
    print(f"Bad summary: {score_summary(test_summary_bad, test_prompt):.2f}")
    print(f"Too long summary: {score_summary(test_summary_too_long, test_prompt):.2f}")
