# MultiAgentsWithDecidingAgent
Masters Degree Dissertation
В последние годы большие языковые модели (LLM) продемонстрировали значительные достижения в области обработки естественного языка, выполняя разнообразные задачи, такие как генерация текста, понимание контекста и логическое рассуждение. Эти успехи были достигнуты благодаря увеличению масштабов моделей, улучшению архитектур и использованию огромных объемов данных для их обучения. Однако в современных реалиях не всегда можно повысить качество генерации LLM за счет дообучения модели из-за требовательности к ресурсам и данным.

Недавнее исследование в работе <<More Agents Is All You Need>>, показало, что производительность LLM может значительно улучшиться за счет простого увеличения числа агентов. Исследование продемонстрировало, что использование метода выборки и голосования с большим числом агентов приводит к существенному повышению точности выполнения задач, даже превосходя результаты более крупных моделей.

В данной работе проанализирован подход ансамблирования LLM агентов для решения различных задач, а также предложена своя интерпретация с заменой голосования на принятие решения финальным агентом (Deciding-Agent). Этот агент анализирует решения, предоставленные основной мультиагентной системой, и выбирает наиболее оптимальный ответ.

В первой главе будут показаны результаты проверки эффективности метода на доменах истории(знание фактологии) и математики(умение рассуждать). Для этого будет проведен сравнительный анализ 4 конфигураций: Один Агент (few-shot-prompting + CoT), Мультиагентная система с голосованием (few-shot-prompting + CoT + majority-voting), Мультиагентная система с решающим агентом без CoT(few-shot-prompting + CoT + Deciding-Agent), Мультиагентная система с решающим агентом с CoT(few-shot-prompting + CoT + Deciding-Agent with CoT).

Во второй главе будет продемонстрирован новый подход к использованию мульти агентной системы на RuBQ-like датасете совместно с RAG. Будут подробно описаны результаты двух экспериментов, а также приведена интерпретация с помощью тепловой карты Attention на открытых данных.

В третьей главе рассмотрим данный подход для имитации игры в <<Что?Где?Когда?>> с использованием GPT-4, проанализируем результаты на нескольких вопросах.

В рамках данной дипломной работы был предложен инновационный подход к улучшению производительности мультиагентных систем, основанный на использовании дополнительного решающего агента. Данный подход позволил добиться значительного прироста ключевой метрики в решении задач.
    
Введение решающего агента продемонстрировало следующие основные достижения:\\
    1. Повышение точности: Существенное улучшение результатов по ключевой метрике, что свидетельствует о высокой эффективности предложенного подхода.\\
    2. Универсальность метода: Возможность применения дополнительного решающего агента к различным задачам и моделям, что делает метод универсальным и гибким.\\
    3. Синергия с существующими методами: Совместимость и улучшение результатов при сочетании с другими сложными методами, такими как цепочки размышлений (CoT).
