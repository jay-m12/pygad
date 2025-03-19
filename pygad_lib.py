# mutation 부분 완전 랜덤

import os
import pygad
import numpy as np
import random
import time

NUM_GENERATIONS = 1000
SOL_PER_POP = 1000
NUM_PARENTS_MATING = 1000
GENE_SPACE_HIGH = None
ELITISM =800
CHANGE_AMOUNT = 0.2
MUTATION_PROB = 0.1
KEEP_PARENTS = 0
PARENT_SELECTION_TYPE = 'rws'
MUTATION_TYPE = 'random'
CROSSOVER_TYPE = "single_point"
INPUT_DIRECTORY = "/home/dna/people/doyoung/PYGAD_DATA/BLASTP_OUTPUT"
OUTPUT_DIRECTORY = "/home/dna/people/jiyoung/output"


def fitness_function(solution, solution_idx, pad_sequence2, ga_instance):  
    # fitness_score = sum([1 if solution[i] == pad_sequence2[i] else 0 for i in range(len(pad_sequence2))])

    # if ga_instance.generations_completed == 0:  
    #     ga_instance.solutions = []  
    # ga_instance.solutions.append(fitness_score)  

    # return fitness_score
    fitness_score = 10   # fitness_score이 0으로 출력될 경우, 오류로 코드가 작동하지 않음. 
    return fitness_score

def make_initial_population(seq, sol_per_pop, change_amount=CHANGE_AMOUNT):   #ex) 0.2 = 전체 중 20퍼센트를 변경
    initial_population_ = []
    target_length = len(seq)

    change_count = max(1, int(target_length * change_amount))
    possible_numbers = list(set(seq))

    for _ in range(sol_per_pop):
        result = list(seq)
        indices_to_change = random.sample(range(target_length), change_count)

        for idx in indices_to_change:
            current_val = result[idx]
            available_vals = [val for val in possible_numbers if val != current_val]
            result[idx] = random.choice(available_vals)

        initial_population_.append(result)

    initial_population = np.array(initial_population_).reshape(sol_per_pop, -1)
    return initial_population




def on_generation(ga_instance):
    # 첫 번째 조건: 최대 세대 수에 도달했을 때
    if ga_instance.generations_completed >= ga_instance.num_generations:
        print(f"Stopping evolution at generation {ga_instance.generations_completed} (max generations reached).")
        ga_instance.generations_completed = ga_instance.num_generations  

    # 두 번째 조건: stop_criteria에 맞는 조건이 만족됐을 때 (목표 길이를 찾았을 때)
    if ga_instance.solutions and 'reach_' + str(len(ga_instance.solutions)) in ga_instance.stop_criteria:
        print(f"Stopping evolution at generation {ga_instance.generations_completed} (target sequence reached).")
        ga_instance.generations_completed = ga_instance.num_generations  # 강제 종료

    # 로그 출력
    else:
        ga_instance.logger.info("Generation    = {generation}".format(generation=ga_instance.generations_completed))
        ga_instance.logger.info("Best Fitness_score = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))

def run_ga(sequence, num_generations, sol_per_pop, output_path, sequence_id, num_parents_mating, gene_space_high):
    pad_sequence2 = sequence

    initial_population = make_initial_population(sequence, sol_per_pop)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        fitness_func=lambda ga, sol, sol_idx: fitness_function(sol, sol_idx, pad_sequence2, ga),
        num_genes=len(pad_sequence2),
        gene_space={"low": 1, "high": gene_space_high},
        crossover_type=CROSSOVER_TYPE,
        mutation_type=MUTATION_TYPE,
        initial_population=initial_population,
        parent_selection_type=PARENT_SELECTION_TYPE,
        keep_parents=KEEP_PARENTS,
        mutation_probability=MUTATION_PROB,
        keep_elitism=ELITISM,
        gene_type=int,
        stop_criteria=[f"reach_{len(sequence)}"],
        parallel_processing=None,
        on_generation=on_generation
    )

    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    if best_solution_fitness == len(pad_sequence2):
        return True
    else:
        print(f"Target sequence {sequence_id} could not be reached through evolution.")
        with open(output_path, "a") as f:
            f.write(f"{sequence_id}\n")
        return False


def compare_sequences(directory_path, output_directory):
    allowed_extensions = (".fa", ".txt")
    sequence_files = [f for f in os.listdir(directory_path) if f.endswith(allowed_extensions)]
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, "pygad_output.txt")

    all_reachable = True
    for idx, seq_file in enumerate(sequence_files, start=1):
        input_path = os.path.join(directory_path, seq_file)
        print(f"\nProcessing file: {seq_file} ({len(sequence_files) - idx} remaining)...")

        with open(input_path, 'r') as f:
            lines = f.readlines()
            if not lines[0].startswith(">"):
                raise ValueError(f"Invalid FASTA format in {seq_file}")

            sequence_id = lines[0].strip()[1:]
            raw_sequence = ''.join(lines[1:]).replace('\n', '').strip()
            char_list = list(raw_sequence)

            custom_mapping = {}
            current_number = [1]
            for char in char_list:
                if char not in custom_mapping:
                    custom_mapping[char] = current_number[0]
                    current_number[0] += 1
            numeric_sequence = [custom_mapping[char] for char in char_list]

        print(f"Sequence ID: {sequence_id}")
        # print(f"Char List (first 10): {char_list}")
        # print(f"Numeric Seq (first 10): {numeric_sequence}")
        print(f"Mapping len: {len(custom_mapping)}")

        gene_space_high = len(custom_mapping) + 1
        reachable = run_ga(
            numeric_sequence,
            NUM_GENERATIONS,
            SOL_PER_POP,
            output_path,
            sequence_id,
            NUM_PARENTS_MATING,
            gene_space_high
        )
        if not reachable:
            all_reachable = False

    if all_reachable:
        print("All sequences could evolve to match themselves. No output file generated.")
    else:
        print(f"Results have been saved to {output_path}.")


def main():
    start_time = time.time()  

    directory_path = INPUT_DIRECTORY
    output_directory = OUTPUT_DIRECTORY
    try:
        compare_sequences(directory_path, output_directory)
    except Exception as e:
        print(f"Error during execution: {e}")

    end_time = time.time()  
    elapsed_time = end_time - start_time  

    print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds")

main()
