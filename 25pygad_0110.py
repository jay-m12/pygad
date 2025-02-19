# 파일에서 시퀀스 읽기
import os
import pygad
import numpy as np
import random
import bisect
import pdb


def load_sequences_from_directory(directory_path, sequence1_file_name):
    """
    지정된 sequence1 파일과 디렉토리 내의 나머지 파일을 로드합니다.
    """
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    if len(files) < 2:
        raise ValueError("디렉토리에는 최소한 두 개의 .txt 파일이 필요합니다.")
    
    # sequence1 파일 읽기
    if sequence1_file_name not in files:
        raise ValueError(f"sequence1 파일 {sequence1_file_name}이 디렉토리에 없습니다.")
    
    with open(os.path.join(directory_path, sequence1_file_name), 'r') as f:
        sequence1 = ''.join(f.readlines()[1:]).replace('\n', '').strip()

    # 나머지 타겟 파일들 처리
    target_files = [f for f in files if f != sequence1_file_name]
    target_sequences = []
    for file in target_files:
        with open(os.path.join(directory_path, file), 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # 두 번째 줄부터 내용이 있는 경우
                target_sequences.append(''.join(lines[1:]).replace('\n', '').strip())
            else:
                raise ValueError(f"파일 {file}에 두 번째 줄부터 내용이 없습니다.")
    
    return sequence1, target_sequences, target_files

def map_char_to_number(char, custom_mapping, current_number):
    if char == 'A':
        return 1
    elif char == 'C':
        return 2
    elif char == 'G':
        return 3
    elif char == 'T':
        return 4
    else:
        if char not in custom_mapping:
            custom_mapping[char] = current_number[0]
            current_number[0] += 1
        return custom_mapping[char]
    
ga_pop = []

def fitness_function(solution, solution_idx, pad_sequence2, ga_instance):
    """
    각 숫자 비교해 같은 값을 가지면 +1, 아니면 0
    """
    # pad_sequence2가 배열 형태인지를 확인
    if not isinstance(pad_sequence2, (list, np.ndarray)):
        raise TypeError("pad_sequence2 must be a list or np.ndarray")
    
    fitness_score = sum([1 if solution[i] == pad_sequence2[i] else 0 for i in range(len(pad_sequence2))])

    if ga_instance.generations_completed == 0:  # 첫 번째 세대에만 추가 (초기화 시)
        ga_instance.solutions = []  # 초기화
    ga_instance.solutions.append(fitness_score)  # solutions 배열에 적합도 값 추가
    
    
    #print(f"Fitness for solution {solution_idx}: {fitness_score}")  # 추가된 출력문 -> 잘 출력됨 확인

    return fitness_score

def replace_random_characters(input_str, num_replacements):
    result = list(input_str)
    indices_to_replace = random.sample(range(len(result)), num_replacements)
    #print(indices_to_replace)

    for idx in indices_to_replace:
        current_char = result[idx]
        possible_chars = ["A", "T", "C", "G"]
        possible_chars.remove(current_char)
        new_char = random.choice(possible_chars)
        result[idx] = new_char

    return "".join(result)

def make_initial_population(seq, sol_per_pop, change_count, custom_mapping, current_number):
    """
    초기 개체군 생성: 타겟 시퀀스의 1/3을 랜덤으로 변경.
    """
    initial_population_ = []
    target_length = len(seq)
    change_count = max(1, target_length // 3)  # 타겟 시퀀스 길이의 1/3 계산 (최소 1)

    for _ in range(sol_per_pop):
        # 랜덤 위치를 선택하여 변경
        result_str = list(seq)
        indices_to_change = random.sample(range(target_length), change_count)  # 1/3 위치 랜덤 선택
        for idx in indices_to_change:
            current_char = result_str[idx]
            possible_chars = ["A", "T", "C", "G"]
            possible_chars.remove(current_char)  # 자신을 제외한 문자
            result_str[idx] = random.choice(possible_chars)  # 랜덤 변경

        # 숫자 매핑
        numeric = [map_char_to_number(char, custom_mapping, current_number) for char in result_str]
        initial_population_.append(numeric)

    initial_population = np.array(initial_population_).reshape(sol_per_pop, -1)

    print("Initial population shape (after creation):", initial_population.shape)  # 확인용 출력
    if initial_population.shape[0] == 0:
        print("Warning: Initial population is empty!")  # 초기 개체군이 비어있다면 경고

    return initial_population


def weighted_sampler(seq, weights):
    #print("sampler start")
    totals = []
    for w in weights:
        if totals:
            totals.append(w + totals[-1])
        else:
            totals.append(w)
    #print(totals)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    fitness_scores = ga_instance.last_generation_fitness
    sampler = weighted_sampler(ga_instance.population, fitness_scores)
    while len(offspring) != offspring_size[0]:
        parents = np.array([sampler() for _ in range(2)])
        #print(parents)
        #print("_____select_____")
        parent1 = parents[0].copy()
        parent2 = parents[1].copy()
        #############################################################################
        # cross over
        random_split_point = np.random.choice(range(offspring_size[1]))
        #print(random_split_point)
        #print("___parent___")
        #print(parent1, parent2)
        #print("___")
        result = np.concatenate((parent1[:random_split_point], parent2[random_split_point:]))
        #print("result")
        #print(result)
        offspring.append(result)
    #print(offspring)
        
    return np.array(offspring)

def parent_selection_func(fitness, num_parents, ga_instance):
    # 현재 세대의 개체 수
    num_population = ga_instance.population.shape[0]
    # select all population to parents
    selected_indices = np.arange(num_population)
    parents = ga_instance.population[selected_indices].copy()
    #print(parents)
    
    return parents, selected_indices

def on_generation(ga_instance):
    # 첫 번째 조건: 최대 세대 수에 도달했을 때
    if ga_instance.generations_completed >= ga_instance.num_generations:
        print(f"Stopping evolution at generation {ga_instance.generations_completed} (max generations reached).")
        ga_instance.stop()  # 진화 멈추기

    # 두 번째 조건: stop_criteria에 맞는 조건이 만족됐을 때 (목표 길이를 찾았을 때)
    if ga_instance.solutions and 'reach_' + str(len(ga_instance.solutions)) in ga_instance.stop_criteria:
        print(f"Stopping evolution at generation {ga_instance.generations_completed} (target sequence reached).")
        ga_instance.stop()  # 진화 멈추기

    # 진화 중에 로그 출력
    else:
        ga_instance.logger.info("Generation    = {generation}".format(generation=ga_instance.generations_completed))
        ga_instance.logger.info("Fitness_score = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        ga_instance.logger.info("best_solution = {solution}".format(solution=ga_instance.best_solution()[0]))


def mutation_func(offspring, ga_instance, pad_sequence2):
    base_characters = [1, 2, 3, 4, 5]
    print('mutation start')
    
    new_offspring_fitness = []  

    for chromosome_idx in range(offspring.shape[0]):
        
        # Mutation: 2개의 랜덤한 값 변경 (기존 값과 다르게)
        random_gene_indices = np.random.choice(range(offspring.shape[1]), size=2, replace=False)
        
        for random_gene_idx in random_gene_indices:
            current_gene = offspring[chromosome_idx, random_gene_idx]
            available_characters = [c for c in base_characters if c != current_gene]  # 현재 값 제외
            new_gene = np.random.choice(available_characters)  # 새로운 값 선택
            offspring[chromosome_idx, random_gene_idx] = new_gene

        ###### Keep `5` at the end ######
        num_fives = np.sum(offspring[chromosome_idx] == 5)
        non_five_genes = offspring[chromosome_idx][offspring[chromosome_idx] != 5]
        offspring[chromosome_idx] = np.concatenate((non_five_genes, np.full(num_fives, 5)))

        ##################################### fitness #####################################
        offspring_fitness = fitness_function(offspring[chromosome_idx], chromosome_idx, pad_sequence2, ga_instance)  # 수정된 부분
        new_offspring_fitness.append(offspring_fitness)

    # Combine previous generation and new offspring by fitness
    last_population = ga_instance.population
    last_population_fitness = ga_instance.last_generation_fitness.tolist()

    combined_list = [(value, index, list_num) 
                     for list_num, lst in enumerate([last_population_fitness, new_offspring_fitness]) 
                     for index, value in enumerate(lst)]
    combined_list.sort(reverse=True)
    
    top_elements = combined_list[:ga_instance.sol_per_pop - 1]
    
    result = []
    for value, index, list_num in top_elements:
        if list_num == 0:
            result.append(last_population[index].tolist())
        elif list_num == 1:
            result.append(offspring[index].tolist())
    
    final_offspring = np.array(result)
    print(final_offspring)

    return final_offspring



def run_ga(sequence1, target_sequence):
    custom_mapping = {}
    current_number = [5]

    # 패딩된 시퀀스 생성: sequence1 길이에 맞춤
    pad_sequence = target_sequence + ("O" * max(0, (len(sequence1) - len(target_sequence))))
    pad_sequence = pad_sequence[:len(sequence1)]  # 길이 초과 방지
    pad_sequence2 = [map_char_to_number(char, custom_mapping, current_number) for char in pad_sequence]

    # 초기 개체군 생성: sequence1 길이에 맞춤
    sol_per_pop = 1000
    initial_population = make_initial_population(sequence1, sol_per_pop, 5, custom_mapping, current_number)
    if initial_population.shape[1] != len(pad_sequence2):
        raise ValueError("Initial population and pad_sequence2 must have the same length.")
    
    num_generations = 35000     # stop 기준1
    num_parents_mating = 1000
    gene_space_high = len(pad_sequence2) + 1

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=lambda ga_instance, solution, solution_idx: fitness_function(solution, solution_idx, pad_sequence2, ga_instance),
                           sol_per_pop=sol_per_pop,
                           num_genes=len(pad_sequence2),
                           gene_space={"low": 1, "high": gene_space_high},
                           parent_selection_type=parent_selection_func,
                           keep_parents=0,
                           crossover_type=crossover_func,
                           mutation_type=lambda offspring, ga_instance: mutation_func(offspring, ga_instance, pad_sequence2),
                           mutation_probability=0.07,
                           gene_type=int,
                           initial_population=initial_population,
                           keep_elitism=1,
                           on_generation=on_generation,
                           stop_criteria=[f"reach_{len(sequence1)}"])
    
    ga_instance.run()

    return ga_instance

def get_first_file_as_sequence1(directory_path):
    """
    디렉토리 내 파일 중 첫 번째 파일을 sequence1으로 선택합니다.
    """
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    if not files:
        raise ValueError("디렉토리에 .txt 파일이 없습니다.")
    return files[0]

def compare_sequences(directory_path, sequence1_file_name):
    """
    sequence1과 디렉토리 내 나머지 파일을 비교합니다.
    """
    sequence1, target_sequences, target_files = load_sequences_from_directory(directory_path, sequence1_file_name)

    # 각 타겟 시퀀스와 순차적으로 비교
    for i, (target_sequence, target_file) in enumerate(zip(target_sequences, target_files), start=1):
        print(f"\nComparing sequence1 from file {sequence1_file_name} with target sequence {i} from file {target_file}...")
        
        ga_instance = run_ga(sequence1, target_sequence)
        
        print(f"Finished comparison of sequence1 from file {sequence1_file_name} with target sequence {i} from file {target_file}.")
        print("Best solution:", ga_instance.best_solution())

        # 다음 타겟으로 진화 시작
        print(f"Starting next target sequence comparison...\n")

def main():
    directory_path = "/home/dna/people/doyoung/pygad/Intergenic_sequence list"  # 디렉토리 경로
    try:
        sequence1_file_name = get_first_file_as_sequence1(directory_path)  # 첫 번째 파일 선택
        compare_sequences(directory_path, sequence1_file_name)
    except Exception as e:
        print(f"Error during execution: {e}")

# 실행
main()