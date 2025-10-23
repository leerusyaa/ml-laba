import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('your_survey_data.csv')  # замените на ваш файл

# Создание признака количества осенних курсов
def get_fall_course_num(row):
    group = row['group']
    # Специализации МОП (61, 62) и ТИ (63) - 2 курса
    if group in ['61', '62', '63']:
        return 2
    # Специализация МИ (второй курс) - 2 курса
    elif str(group).startswith('2'):  # второй курс
        # Уточните условие для МИ специализации
        return 2
    else:
        return 1

df['fall_course_num'] = df.apply(get_fall_course_num, axis=1)
# Ограничения мест на осенние курсы
fall_capacity = {
    'Statistical Learning Theory': 60,
    'Высокопроизводительные вычисления': 60,
    'Анализ неструктурированных данных': float('inf')
}

# По умолчанию 30 мест для остальных курсов
default_capacity = 30


def distribute_students(df, season='fall'):
    """
    Распределение студентов по курсам
    """
    # Определяем приоритетные колонки в зависимости от сезона
    if season == 'fall':
        priority_cols = ['fall_1', 'fall_2', 'fall_3']
        course_num_col = 'fall_course_num'
    else:
        priority_cols = ['spring_1', 'spring_2', 'spring_3']
        course_num_col = 'spring_course_num'

    # Инициализация результатов
    results = {}

    # Создаем словарь для отслеживания оставшихся мест
    capacity = fall_capacity.copy()
    all_courses = set()
    for col in priority_cols:
        all_courses.update(df[col].unique())

    for course in all_courses:
        if course not in capacity:
            capacity[course] = default_capacity

    # Волна 1: первые приоритеты
    wave1_results = process_wave(df, capacity, priority_cols[:2], wave_num=1, course_num_col=course_num_col)
    update_capacity(capacity, wave1_results)

    # Волна 2: вторые приоритеты
    wave2_results = process_wave(df, capacity, [priority_cols[2]], wave_num=2, course_num_col=course_num_col,
                                 existing_assignments=wave1_results)
    update_capacity(capacity, wave2_results)

    # Объединяем результаты
    final_results = combine_results(wave1_results, wave2_results)

    return final_results, capacity


def process_wave(df, capacity, priority_cols, wave_num, course_num_col, existing_assignments=None):
    """
    Обработка одной волны распределения
    """
    if existing_assignments is None:
        existing_assignments = {}

    wave_results = existing_assignments.copy()

    for course in capacity:
        if capacity[course] <= 0:
            continue

        # Собираем кандидатов на курс
        candidates = []
        for student_id, row in df.iterrows():
            # Проверяем, нужны ли студенту еще курсы
            current_courses = wave_results.get(student_id, [])
            needed_courses = row[course_num_col] - len(current_courses)

            if needed_courses <= 0:
                continue

            # Проверяем приоритеты
            for priority, pref_col in enumerate(priority_cols):
                if row[pref_col] == course:
                    candidates.append({
                        'student_id': student_id,
                        'priority': priority,
                        'percentile': row['percentile'],
                        'needed_courses': needed_courses
                    })
                    break

        # Сортируем кандидатов по перцентилю
        candidates.sort(key=lambda x: x['percentile'], reverse=True)

        # Распределяем места
        assigned = 0
        for candidate in candidates:
            if assigned >= capacity[course]:
                break

            student_id = candidate['student_id']
            if student_id not in wave_results:
                wave_results[student_id] = []

            # Проверяем, что студент еще не записан на этот курс
            if course not in wave_results[student_id]:
                wave_results[student_id].append(course)
                assigned += 1

    return wave_results


def create_final_dataframe(df, distribution_results, season='fall'):
    """
    Создание финального DataFrame в требуемом формате
    """
    results_list = []

    for student_id, row in df.iterrows():
        assigned_courses = distribution_results.get(student_id, [])
        course_num_needed = row['fall_course_num'] if season == 'fall' else row['spring_course_num']

        # Заполняем курсы
        course1 = assigned_courses[0] if len(assigned_courses) > 0 else "???"
        course2 = assigned_courses[1] if len(assigned_courses) > 1 else ("???" if course_num_needed > 1 else "-")

        # Если нужно 2 курса, но распределен только 1
        if course_num_needed == 2 and len(assigned_courses) == 1:
            course2 = "???"

        results_list.append({
            'ID': student_id,
            'course1': course1,
            'course2': course2
        })

    return pd.DataFrame(results_list)


# Основной процесс распределения
fall_distribution, remaining_capacity = distribute_students(df, 'fall')

# Создание финального файла
fall_results_df = create_final_dataframe(df, fall_distribution, 'fall')
fall_results_df.to_csv('res_fall.csv', index=None)

print("Распределение на осенние курсы завершено!")


# Проверка специальных случаев (МОП и Машинное обучение 2)
def fix_mop_preferences(df):
    """
    Корректировка приоритетов для студентов МОП
    """
    mop_students = df[df['group'].isin(['61', '62'])]

    for idx, row in mop_students.iterrows():
        if 'Машинное обучение 2' in [row['spring_1'], row['spring_2']]:
            # Сдвигаем приоритеты весенних курсов
            # Реализуйте логику сдвига приоритетов
            pass

    return df


# Применяем корректировки
df = fix_mop_preferences(df)
