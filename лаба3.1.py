import pandas as pd
import numpy as np

# Открываем файл и выводим последние 10 строк
df = pd.read_csv('./data/lab01_train.csv')
print("Последние 10 строк:")
print(df.tail(10))
print("\nИнформация о данных:")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")
print("Столбцы:", df.columns.tolist())

# 1. Сколько заявок от студентов второго и третьего курса
number_of_request_course_2 = df['course_2'].notna().sum()
number_of_request_course_3 = df['course_3'].notna().sum()

print(f"\nЗаявок от студентов 2 курса: {number_of_request_course_2}")
print(f"Заявок от студентов 3 курса: {number_of_request_course_3}")

# 2. Есть ли студенты с равными перцентилями
is_equal_percentil = df.duplicated(subset=['percentile']).any()

print(f"\nЕсть студенты с равными перцентилями: {is_equal_percentil}")

# 3. Пропуски в данных
course_2_na = df['course_2'].isna().sum()
course_3_na = df['course_3'].isna().sum()
is_mi_student_na = df['is_mi_student'].isna().sum()
is_ml_student_na = df['is_ml_student'].isna().sum()
is_first_time_na = df['is_first_time'].isna().sum()
blended_na = df['blended'].isna().sum()
percentile_na = df['percentile'].isna().sum()

print(f"\nПропуски по столбцам:")
print(f"course_2: {course_2_na}")
print(f"course_3: {course_3_na}")
print(f"is_mi_student: {is_mi_student_na}")
print(f"is_ml_student: {is_ml_student_na}")
print(f"is_first_time: {is_first_time_na}")
print(f"blended: {blended_na}")
print(f"percentile: {percentile_na}")

# 4. Заполнение пропусков
df_filled = df.copy()
# Строковые колонки
string_columns = ['course_2', 'course_3', 'fall_1', 'fall_2', 'fall_3',
                  'spring_1', 'spring_2', 'spring_3', 'blended', 'is_first_time']
# Числовые колонки
numeric_columns = ['is_mi_student', 'is_ml_student', 'percentile']

# Заполняем строковые колонки пустой строкой
df_filled[string_columns] = df_filled[string_columns].fillna('')
# Заполняем числовые колонки нулем
df_filled[numeric_columns] = df_filled[numeric_columns].fillna(0)

# Проверяем, что пропусков больше нет
course_2_na = df_filled['course_2'].isna().sum()
course_3_na = df_filled['course_3'].isna().sum()
is_mi_student_na = df_filled['is_mi_student'].isna().sum()
is_ml_student_na = df_filled['is_ml_student'].isna().sum()
is_first_time_na = df_filled['is_first_time'].isna().sum()
blended_na = df_filled['blended'].isna().sum()

df_na = df_filled.isna().sum().sum()

print(f"\nПосле заполнения пропусков:")
print(f"Общее количество пропусков: {df_na}")

# 5. Количество ответов "Нет" в колонке is_first_time
number_is_first_time_responses_no = (df_filled['is_first_time'] == 'Нет').sum()
print(f"\nКоличество ответов 'Нет' в is_first_time: {number_is_first_time_responses_no}")

# 6. Удаление повторных обращений (оставляем только самые поздние)
# Сортируем по id и дате (предполагая, что есть временная метка или порядок)
# Если временной метки нет, оставляем первую запись для каждого id
df_unique = df_filled.drop_duplicates(subset=['id'], keep='first')
total_number_of_request = df_unique.shape[0]

print(f"\nПосле удаления повторов осталось заявок: {total_number_of_request}")

# 7. Blended-курсы для второкурсников
blended_courses_for_second_year_students = set(
    df_unique[df_unique['course_2'].notna() & (df_unique['course_2'] != '')]['blended'].unique())
# Убираем пустые строки если есть
blended_courses_for_second_year_students = {x for x in blended_courses_for_second_year_students if x != ''}

print(f"\nBlended-курсы для второкурсников: {blended_courses_for_second_year_students}")

# 8. Blended-курс с наибольшим количеством студентов
blended_counts = df_unique['blended'].value_counts()
# Исключаем пустую строку
blended_counts = blended_counts[blended_counts.index != '']
blended_course_with_max_request = blended_counts.index[0] if not blended_counts.empty else None

print(f"\nBlended-курс с наибольшим количеством студентов: {blended_course_with_max_request}")

# 9. Курс с самым высоким средним рейтингом
# Создаем список всех курсов
all_courses = []
for col in ['fall_1', 'fall_2', 'fall_3', 'spring_1', 'spring_2', 'spring_3']:
    course_percentiles = df_unique.groupby(col)['percentile'].mean()
    all_courses.append(course_percentiles)

# Объединяем все курсы
all_courses_combined = pd.concat(all_courses)
# Находим курс с максимальным средним перцентилем
course_with_highest_average_rating = all_courses_combined.idxmax()

print(f"\nКурс с самым высоким средним рейтингом: {course_with_highest_average_rating}")

# 10-11. Дублирующиеся наборы курсов
# Создаем столбец с комбинацией всех выбранных курсов
df_unique['courses_combination'] = df_unique['fall_1'] + '|' + df_unique['fall_2'] + '|' + df_unique['fall_3'] + '|' + \
                                   df_unique['spring_1'] + '|' + df_unique['spring_2'] + '|' + df_unique[
                                       'spring_3'] + '|' + \
                                   df_unique['blended']

# Считаем количество студентов для каждой комбинации
combination_counts = df_unique['courses_combination'].value_counts()
# Находим комбинации, которые выбрали более 1 студента
duplicate_combinations = combination_counts[combination_counts > 1]
number_duplicate_sets_of_courses = len(duplicate_combinations)

print(f"\nКоличество дублирующихся наборов курсов: {number_duplicate_sets_of_courses}")

# Создаем DataFrame с дублирующимися наборами
duplicate_sets = []
for combination, count in duplicate_combinations.items():
    if count > 1:
        # Берем первую запись с этой комбинацией для получения названий курсов
        sample_row = df_unique[df_unique['courses_combination'] == combination].iloc[0]
        duplicate_sets.append({
            'fall_1': sample_row['fall_1'],
            'fall_2': sample_row['fall_2'],
            'fall_3': sample_row['fall_3'],
            'spring_1': sample_row['spring_1'],
            'spring_2': sample_row['spring_2'],
            'spring_3': sample_row['spring_3'],
            'blended': sample_row['blended'],
            'count': count
        })

duplicate_sets_of_courses_with_number_students = pd.DataFrame(duplicate_sets)

print(f"\nДублирующиеся наборы курсов:")
print(duplicate_sets_of_courses_with_number_students)

# 12. Курсы, на которые записывались и второкурсники и третьекурсники
# Получаем все уникальные курсы
all_fall_courses = set(pd.concat([df_unique['fall_1'], df_unique['fall_2'], df_unique['fall_3']]).unique())
all_spring_courses = set(pd.concat([df_unique['spring_1'], df_unique['spring_2'], df_unique['spring_3']]).unique())
all_courses_set = all_fall_courses.union(all_spring_courses)
# Убираем пустые строки
all_courses_set = {x for x in all_courses_set if x != ''}

courses_students_of_2_and_3_year = set()

for course in all_courses_set:
    # Студенты 2 курса, выбравшие этот курс
    course_2_students = df_unique[
        ((df_unique['fall_1'] == course) | (df_unique['fall_2'] == course) | (df_unique['fall_3'] == course) |
         (df_unique['spring_1'] == course) | (df_unique['spring_2'] == course) | (df_unique['spring_3'] == course)) &
        (df_unique['course_2'] != '')
        ]

    # Студенты 3 курса, выбравшие этот курс
    course_3_students = df_unique[
        ((df_unique['fall_1'] == course) | (df_unique['fall_2'] == course) | (df_unique['fall_3'] == course) |
         (df_unique['spring_1'] == course) | (df_unique['spring_2'] == course) | (df_unique['spring_3'] == course)) &
        (df_unique['course_3'] != '')
        ]

    if not course_2_students.empty and not course_3_students.empty:
        courses_students_of_2_and_3_year.add(course)

print(f"\nКурсы, выбранные и 2 и 3 курсом: {courses_students_of_2_and_3_year}")

# 13. Курсы только для 2 курса и только для 3 курса
courses_only_2_year = set()
courses_only_3_year = set()

for course in all_courses_set:
    course_2_students = df_unique[
        ((df_unique['fall_1'] == course) | (df_unique['fall_2'] == course) | (df_unique['fall_3'] == course) |
         (df_unique['spring_1'] == course) | (df_unique['spring_2'] == course) | (df_unique['spring_3'] == course)) &
        (df_unique['course_2'] != '')
        ]

    course_3_students = df_unique[
        ((df_unique['fall_1'] == course) | (df_unique['fall_2'] == course) | (df_unique['fall_3'] == course) |
         (df_unique['spring_1'] == course) | (df_unique['spring_2'] == course) | (df_unique['spring_3'] == course)) &
        (df_unique['course_3'] != '')
        ]

    if not course_2_students.empty and course_3_students.empty:
        courses_only_2_year.add(course)
    elif course_2_students.empty and not course_3_students.empty:
        courses_only_3_year.add(course)

print(f"\nКурсы только для 2 курса: {courses_only_2_year}")
print(f"Курсы только для 3 курса: {courses_only_3_year}")