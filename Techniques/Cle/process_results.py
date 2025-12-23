import json
import pprint
import argparse
import re

"""
Read filename and max batch number from commandline. Rename all the files to have a single name and number.
"""

parser = argparse.ArgumentParser(description='Process Semgrep results.')
parser.add_argument('json_filename', type=str, help='Base filename for Semgrep JSON results')
parser.add_argument('max_batch_num', type=int, help='Maximum batch number to process')

args = parser.parse_args()

json_filename = args.json_filename
max_batch_num = args.max_batch_num
multiple_samples = False

"""
Read json file. Count number of issues, number of scanned file, number of files that caused errors and compute issues percentage.
"""

total_errors = []
total_results = []
total_scanned = []

for i in range(0, max_batch_num):
    json_filename_complete=(f"{json_filename}_{i+1}.json")
    with open(json_filename_complete, 'r', encoding='utf-8') as results_f:
        samples = json.load(results_f)
        total_errors.extend(samples['errors'])
        total_results.extend(samples['results'])
        total_scanned.extend(samples['paths']['scanned'])

"""
NB: do this only if the dataset is splitted in multiple batches. 
For each filepath in each batch, find the corresponding dataset line number.
The formula is: ((batch-1)*500,000+i)

- sample_1_1 				-> 1
- …
- sample_1_500000 		    -> 500,000

- sample_2_1 				-> 500,001
- …
- sample_2_500000 		    -> 1,000,000

- sample_3_1				-> 1,000,001
- …
- sample_3_500000		    -> 1,500,000
"""

if multiple_samples:

    pattern = r'\w+_(\d+)_(\d+)\.py'

    # def calculate_line_number(filename):
    #     match = re.match(pattern, filename)
    #     # print(filename)
    #     if match:
    #         batch = int(match.group(1))
    #         sample_number = int(match.group(2))
    #         line_number = ((batch - 1) * 500000) + sample_number
    #         # print(line_number)
    #         return line_number
    #     else:
    #         return None
    def calculate_line_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'(\d+)\.py', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"[Warning] Could not extract line number from: {filepath}")
            return None
else:

    pattern = r'\w+_(\d+)\.py'

    def calculate_line_number(filename):
        match = re.match(pattern, filename)
        print(filename)
        if match:
            line_number = int(match.group(1))
            print(line_number)
            return line_number
        else:
            return None

for error in total_errors:
    new_error_path = calculate_line_number(error['path'])
    error['path'] = new_error_path

for result in total_results:
    new_result_path = calculate_line_number(result['path'])
    result['path'] = new_result_path

for scanned in total_scanned:
    new_scanned_path = calculate_line_number(scanned)
    scanned = new_scanned_path

"""
Numbers don't add up because the overlap between issues and errors is not complete. i.e., only a fraction of the errors 
also contains issues.
----------
dedup_err is the list of errors w/o duplicates
dedup_res is the list of defective functions (i.e., w/o duplicated issues)
total_results is the list of issues w/o errors
dedup_res_no_errors is the list of defective functions w/o errors
"""

dedup_err = set()
for err in total_errors:
    dedup_err.add(err['path'])

dedup_res = set()
for res in total_results:
    dedup_res.add(res['path'])

dedup_res_no_errors = [res for res in dedup_res if res not in dedup_err]
total_results = [res for res in total_results if res['path'] not in dedup_err]

"""
Extract dataset lines with defects and lines with syntax errors.
Write them on json files to be later extracted from dataset.
Line number -1 because filepaths start from 1 instead of 0.
"""

dedup_res_no_errors_data_lines = []
for dedup_res_line in dedup_res_no_errors:
    dedup_res_no_errors_data_lines.append(dedup_res_line-1)

dedup_err_data_lines = []
for dedup_err_line in dedup_err:
    if dedup_err_line is not None:
        dedup_err_data_lines.append(dedup_err_line - 1)
    else:
        print(f"[Warning] Skipped a dedup_err_line because it is None.")

with open('defective_lines_'+json_filename+'.json', 'w', encoding='utf-8') as def_f:
        json.dump((sorted(dedup_res_no_errors_data_lines)), def_f)

with open('syntax_errors_'+json_filename+'.json', 'w', encoding='utf-8') as syn_f:
        json.dump(sorted(dedup_err_data_lines), syn_f)


"""
Get severity types (e.g., INFO, WARNING, ERROR, etc.).
Get category types (e.g., security, best-practise, etc.)
Filter out uncategorized issues in the correct category (i.e., they are all from bandit rules, hence, they are security issues).
Make sure to have consistent naming between CWEs to not count them as different CWEs. 
"""

severity_types = set()
category_types = set()
issues_not_categorized = []

for result in total_results:

    severity = result['extra'].get('severity')
    if severity:
        severity_types.add(severity)
    
    category = result['extra'].get('metadata', {}).get('category')
    if category:
        category_types.add(category)
    else:
        issues_not_categorized.append(result)

    if result['check_id'] == "gitlab.bandit.B303-6" or result['check_id'] == "gitlab.bandit.B303-4":
        result['extra']['metadata']['cwe'] = "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"


"""
Divide issues based on severity type. 
Divide issues based on category type. 
"""

severity_dict = {severity: [] for severity in severity_types}
category_dict = {category: [] for category in category_types}

for result in total_results:
    severity = result['extra'].get('severity')
    if severity:
        severity_dict[severity].append(result)

    category = result['extra'].get('metadata', {}).get('category')
    if category:
        category_dict[category].append(result)
    else:
        cwes = result['extra'].get('metadata', {}).get('cwe')
        if cwes:
            category_dict["security"].append(result)
    
severity_counts = {severity: len(results) for severity, results in severity_dict.items()}
category_counts = {category: len(results) for category, results in category_dict.items()}

"""
Identify top-5 issues per category.
"""

pattern = r'[^.]+$'

category_issue_names = {category: set() for category in category_types if category != "security"}

for category, results in category_dict.items():
    if category != "security":
        for result in results:
            issue_name_complete = result['check_id']
            issue_name = re.search(pattern, issue_name_complete).group(0)
            category_issue_names[category].add(issue_name)

category_name_dict = {
    category: {issue_name: [] for issue_name in issue_names}
    for category, issue_names in category_issue_names.items()
}

for category, results in category_dict.items():
    if category != "security":
        for result in results:
            issue_name_complete = result['check_id']
            issue_name = re.search(pattern, issue_name_complete).group(0)
            category_name_dict[category][issue_name].append(result)

category_names_counts = {
    category: {issue_name: len(results) for issue_name, results in issue_names.items()}
    for category, issue_names in category_name_dict.items()
}

result_of_issue_names_per_category = ""
for category, issue_names in category_names_counts.items():
    result_of_issue_names_per_category += f"\n---> Category: {category} <---\n"
    sorted_issues = sorted(issue_names.items(), key=lambda x: x[1], reverse=True)
    top_5_issues = sorted_issues[:5]
    total_issues_counts = sum(issue_names.values())
    other_count = sum(count for _, count in sorted_issues[5:])
    
    for issue_name, count in top_5_issues:
        issues_rate = (count / total_issues_counts) * 100
        result_of_issue_names_per_category += f"    {issue_name}: {count} ({issues_rate:.2f}%)\n"
    if other_count > 0:  # Only print "OTHERS" if there are more than 5 issues
        other_rate = (other_count / total_issues_counts) * 100
        result_of_issue_names_per_category += f"    OTHERS: {other_count} ({other_rate:.2f}%)\n"



"""
Divide issues of each severity in different categories.
"""

severity_category_dict = {severity: {category: [] for category in category_types} for severity in severity_types}

for severity, results in severity_dict.items():
    for result in results:
        category = result['extra'].get('metadata', {}).get('category')
        if category:
            severity_category_dict[severity][category].append(result)
        else:
            cwes = result['extra'].get('metadata', {}).get('cwe')
            if cwes:
                category_dict["security"].append(result)

severity_category_counts = {severity: {category: len(results) for category, results in categories.items()}
                            for severity, categories in severity_category_dict.items()}

result_of_percentages_per_severity = ""

for severity, category_counts_sev in severity_category_counts.items():
    result_of_percentages_per_severity += f"\n---> Severity: {severity} <---\n"
    
    total_severity_counts = sum(category_counts_sev.values())
    
    for category, count in category_counts_sev.items():
        category_rate = (count / total_severity_counts) * 100 if total_severity_counts > 0 else 0
        result_of_percentages_per_severity += f"    {category}: {category_rate:.2f}%\n"

"""
Identify top-5 issues per category per severity.
"""

pattern = r'[^.]+$'
issue_names_set = set()

for severity, categories in severity_category_dict.items():
    for category, results in categories.items():
        if (category != "security"):
            for result in results:
                issue_name_complete = result['check_id']
                issue_name = re.search(pattern, issue_name_complete).group(0)
                issue_names_set.add(issue_name)

severity_category_name_dict = {severity: {category: {issue_name: [] for issue_name in issue_names_set} for category in category_types} for severity in severity_types}

for severity, categories in severity_category_dict.items():
    for category, results in categories.items():
        if (category != "security"):
            for result in results:
                issue_name_complete = result['check_id']
                issue_name = re.search(pattern, issue_name_complete).group(0)
                severity_category_name_dict[severity][category][issue_name].append(result)

severity_category_names_counts = {severity: {category: {issue_name: len(results) for issue_name, results in issue_names.items()}
                                            for category, issue_names in categories.items() if category != "security"}
                                  for severity, categories in severity_category_name_dict.items()}


result_of_issue_names = ""
for severity, categories in severity_category_names_counts.items():
    result_of_issue_names += f"\n########## Severity: {severity} ##########\n"
    for category, issue_names in categories.items():
        result_of_issue_names += f"\n---> Category: {category} <---\n"
        sorted_issues = sorted(issue_names.items(), key=lambda x: x[1], reverse=True)
        top_5_issues = sorted_issues[:5]
        other_count = sum(count for _, count in sorted_issues[5:])
        
        for issue_name, count in top_5_issues:
            result_of_issue_names += f"    {issue_name}: {count}\n"
        result_of_issue_names += f"    OTHERS: {other_count}\n"

"""
Filter security-related issues based on CWE, OWASP category, impact level and Semgrep confidence level. 
"""

security_issues = []
cwe_types = set()
owasp_categories = set()
impact_levels = set()
confidence_levels = set()

for result in total_results:
    category = result['extra'].get('metadata', {}).get('category')
    if not category or category == "security":
        security_issues.append(result)

        cwes = result['extra'].get('metadata', {}).get('cwe')
        if cwes:
            if isinstance(cwes, list):
                cwe_types.update(cwes)
            else:
                cwe_types.add(cwes)

        owasps = result['extra'].get('metadata', {}).get('owasp')
        if owasps:
            if isinstance(owasps, list):
                owasp_categories.update(owasps)
            else:
                owasp_categories.add(owasps)

        impact = result['extra'].get('metadata', {}).get('impact')
        impact_levels.add(impact)

        confidence = result['extra'].get('metadata', {}).get('confidence')
        confidence_levels.add(confidence)

"""
Divide security-related issues based on CWE. 
Divide security-related issues based on OWASP category. 
Divide security-related issues based on impact level. 
Divide security-related issues based on Semgrep confidence level. 
"""

cwes_dict = {cwe: [] for cwe in cwe_types}
owasps_dict = {owasp: [] for owasp in owasp_categories}
impact_levels_dict = {impact_level: [] for impact_level in impact_levels}
confidence_levels_dict = {confidence_level: [] for confidence_level in confidence_levels}

for issue in security_issues:
    cwes = issue['extra'].get('metadata', {}).get('cwe')
    if cwes:
        if isinstance(cwes, list):
            for cwe in cwes:
                if cwe in cwes_dict:
                    cwes_dict[cwe].append(issue)
        else:
            if cwes in cwes_dict:
                cwes_dict[cwes].append(issue)

    owasps = issue['extra'].get('metadata', {}).get('owasp')
    if owasps:
        if isinstance(owasps, list):
            for owasp in owasps:
                if owasp in owasps_dict:
                    owasps_dict[owasp].append(issue)
        else:
            if owasps in owasps_dict:
                owasps_dict[owasps].append(issue)

    impact = issue['extra'].get('metadata', {}).get('impact')
    if impact:
        impact_levels_dict[impact].append(issue)

    confidence = issue['extra'].get('metadata', {}).get('confidence')
    if confidence:
        confidence_levels_dict[confidence].append(issue)
    
cwes_counts = {cwe: len(issues) for cwe, issues in cwes_dict.items()}
owasps_counts = {owasp: len(issues) for owasp, issues in owasps_dict.items()}
impact_levels_counts = {impact: len(issues) for impact, issues in impact_levels_dict.items()}
confidence_levels_counts = {confidence: len(issues) for confidence, issues in confidence_levels_dict.items()}


"""
For each CWE, count how many are INFO, how many are WARNING and ERRORS.
"""

cwes_severity_counts = {cwe: {'INFO': 0, 'WARNING': 0, 'ERROR': 0} for cwe in cwe_types}

for cwe, issues in cwes_dict.items():
    for issue in issues:
        severity = issue['extra'].get('severity')
        if severity in cwes_severity_counts[cwe]:
            cwes_severity_counts[cwe][severity] += 1

cwes_total_counts = {cwe: sum(counts.values()) for cwe, counts in cwes_severity_counts.items()}

sorted_cwes_total_counts = sorted(cwes_total_counts.items(), key=lambda x: x[1], reverse=True)

for cwe, total_count in sorted_cwes_total_counts:
    counts = cwes_severity_counts[cwe]
    print(f"{cwe}: INFO: {counts['INFO']}, WARNING: {counts['WARNING']}, ERROR: {counts['ERROR']} (Total: {total_count})")


"""
Identify top 5 CWEs for the security category. 
"""

sorted_cwes = sorted(cwes_counts.items(), key=lambda x: x[1], reverse=True)
top_5_cwes = sorted_cwes[:5]
total_cwes = sum(cwes_counts.values())
other_count = sum(count for _, count in sorted_cwes[5:])
result_of_cwes_counts = "---> Category: security <---\n"
for cwe, count in top_5_cwes:
    cwe_rate = (count / total_cwes) * 100
    result_of_cwes_counts += f"    {cwe}: {count} ({cwe_rate:.2f}%)\n"
if other_count > 0:
    other_cwes_rate = (other_count / total_cwes) * 100
    result_of_cwes_counts += f"    OTHERS: {other_count} ({other_cwes_rate:.2f}%)\n"


"""
Compute percentages of defects, errors and clean functions.
"""

defective_func_rate = (len(dedup_res_no_errors)/len(total_scanned)) * 100
errors_rate = (len(dedup_err)/len(total_scanned)) * 100
issues_rate = (len(total_results)/len(total_scanned)) * 100
severity_rates = {severity: (count / len(total_results)) * 100 for severity, count in severity_counts.items()}
category_rates = {category: (count / len(total_results)) * 100 for category, count in category_counts.items()}
clean_rate = (len(total_scanned)-len(dedup_res_no_errors)-len(dedup_err))/len(total_scanned) * 100

"""
Print all gathered information.
"""

print(f"Total scanned functions: {len(total_scanned)} (100%)")
print(f"Total clean functions: {len(total_scanned)-len(dedup_res_no_errors)-len(dedup_err)} ({clean_rate:.2f}%)")
print(f"Total defective functions (excluding errors): {len(dedup_res_no_errors)} ({defective_func_rate:.2f}%)")
print(f"Total errors: {len(total_errors)}. Errors w/o duplicates: {len(dedup_err)} ({errors_rate:.2f}%)")
print(f"Total issues (considering multiple issues per function and excluding errors): {len(total_results)} ({issues_rate:.2f}%)")

print(f"No of issues per severity type: {severity_counts} and their rates:")
for severity, rate in severity_rates.items():
    print(f"{severity}: {rate:.2f}%")
print(f"\nNo of issues per category type: {category_counts} and their rates:")
for category, rate in category_rates.items():
    print(f"{category}: {rate:.2f}%")

# print(f"Top 5 issues per severity, per category type:")
# print(result_of_issue_names)

print(f"\nTop 5 issues per category type:")
print(result_of_issue_names_per_category)
print(result_of_cwes_counts)


print(f"\nNo of issues, per severity, per category type and their rates:")
pprint.pprint(severity_category_counts)
print(result_of_percentages_per_severity)

# print(f"\nNo of security-related issues per impact level: {impact_levels_counts}\n")
# print(f"No of security-related issues per Semgrep confidence level: {confidence_levels_counts}\n")
# print(f"No of different CWEs: {len(cwe_types)}")
# print(f"No of security-related issues per CWE type:")
# pprint.pprint(cwes_counts)

# print(f"\nNo of different OWASP categories: {len(owasp_categories)}")
# print(f"No of security-related issues per OWASP category:")
# pprint.pprint(owasps_counts)