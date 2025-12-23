import json
import os
import subprocess
import glob
import time
import argparse

def split_json_to_python_files(json_file, output_prefix, lines_per_file=1, files_per_batch=20000):
    start_time = time.time()

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    """
    If the json file is from the original dataset, extract the 'output' column, else load data as is. 
    """

    outputs = []
    for item in data:
        try:
            output = item.get('output')
        except:
            outputs.append(item)
        else:
            outputs.append(output)
      

    total_lines = len(outputs)
    total_files = total_lines // lines_per_file
    total_batches = (total_files + files_per_batch - 1) // files_per_batch

    print(f"Total lines: {total_lines}, Total files: {total_files}, Total batches: {total_batches}")

    split_times = []
    semgrep_times = []
    delete_times = []

    for batch in range(total_batches):
        print(f"Processing batch {batch + 1}/{total_batches}")
        batch_start_index = batch * files_per_batch * lines_per_file
        batch_end_index = min((batch + 1) * files_per_batch * lines_per_file, total_lines)
        batch_outputs = outputs[batch_start_index:batch_end_index]

        num_files = (batch_end_index - batch_start_index) // lines_per_file
        
        batch_split_start = time.time()
        for i in range(num_files):
            start_index = batch_start_index + i * lines_per_file
            end_index = start_index + lines_per_file
            chunk = batch_outputs[start_index - batch_start_index:end_index - batch_start_index]

            output_file = f"{output_prefix}_{start_index+1}.py"
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in chunk:
                    f.write(line)
        batch_split_end = time.time()
        split_times.append(batch_split_end - batch_split_start)
        
        json_filename = f"{output_prefix}_semgrep_results_batch_{batch+1}.json"
        batch_semgrep_time = run_semgrep_analysis(json_filename)
        semgrep_times.append(batch_semgrep_time)

        batch_delete_time = delete_python_files(output_prefix, batch, num_files, lines_per_file)
        delete_times.append(batch_delete_time)

    end_time = time.time()
    split_json_time = end_time - start_time
    return split_json_time, split_times, semgrep_times, delete_times

def run_semgrep_analysis(json_filename):
    start_time = time.time()

    print(f"Running Semgrep analysis and saving results to {json_filename}...")
    semgrep_command = [
        "semgrep", "scan",
        "--verbose",
        "--output", json_filename,
        "--json",
        "--max-memory=30000",
        "--max-target-bytes=1000000",
        "--timeout-threshold", "2",
        "--timeout", "10",
        "--config", "p/trailofbits",
        "--config", "p/default",
        "--config", "p/bandit",
        "--config", "p/comment",
        "--config", "p/python",
        "--config", "p/cwe-top-25",
        "--config", "p/owasp-top-ten",
        "--config", "p/r2c-security-audit",
        "--config", "p/insecure-transport",
        "--config", "p/secrets",
        "--config", "r/python.attr.correctness.mutable-initializer.attr-mutable-initializer",
        "--config", "r/python.bokeh.maintainability.deprecated.deprecated_apis.bokeh-deprecated-apis",
        "--config", "r/python.click.best-practice.echo-style.use-click-secho",
        "--config", "r/python.correctness.socket-shutdown-close.socket-shutdown-close",
        "--config", "r/python.correctness.suppressed-exception-handling-finally-break.suppressed-exception-handling-finally-break",
        "--config", "r/python.django.best-practice.json_response.use-json-response",
        "--config", "r/python.django.best-practice.upsell_django_environ.use-django-environ",
        "--config", "r/python.django.best-practice.use-onetoonefield.use-onetoonefield",
        "--config", "r/python.django.correctness.model-save.django-db-model-save-super",
        "--config", "r/python.django.correctness.nontext-field-must-set-null-true.nontext-field-must-set-null-true",
        "--config", "r/python.django.correctness.string-field-null-checks.no-null-string-field",
        "--config", "r/python.django.correctness.string-field-null-checks.string-field-must-set-null-true",
        "--config", "r/python.django.correctness.use-decimalfield-for-money.use-decimalfield-for-money",
        "--config", "r/python.django.maintainability.duplicate-path-assignment.conflicting-path-assignment",
        "--config", "r/python.django.maintainability.duplicate-path-assignment.duplicate-name-assignment",
        "--config", "r/python.django.maintainability.duplicate-path-assignment.duplicate-path-assignment",
        "--config", "r/python.django.maintainability.duplicate-path-assignment.duplicate-path-assignment-different-names",
        "--config", "r/python.django.performance.access-foreign-keys.access-foreign-keys",
        "--config", "r/python.django.performance.upsell-count.use-count-method",
        "--config", "r/python.django.performance.upsell_earliest_latest.use-earliest-or-latest",
        "--config", "r/python.flask.best-practice.get-class-method-with-side-effects.flask-class-method-get-side-effects",
        "--config", "r/python.flask.best-practice.use-jsonify.use-jsonify",
        "--config", "r/python.flask.correctness.access-request-in-wrong-handler.avoid-accessing-request-in-wrong-handler",
        "--config", "r/python.flask.correctness.same-handler-name.flask-duplicate-handler-name",
        "--config", "r/python.flask.maintainability.deprecated.deprecated-apis.flask-deprecated-apis",
        "--config", "r/python.lang.best-practice.hardcoded-tmp-path.hardcoded-tmp-path",
        "--config", "r/python.lang.best-practice.logging-error-without-handling.logging-error-without-handling",
        "--config", "r/python.lang.best-practice.manual-collections-create.manual-counter-create",
        "--config", "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-dict-create",
        "--config", "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-list-create",
        "--config", "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-set-create",
        "--config", "r/python.lang.best-practice.missing-hash-with-eq.missing-hash-with-eq",
        "--config", "r/python.lang.best-practice.open-never-closed.open-never-closed",
        "--config", "r/python.lang.best-practice.pass-body.pass-body-fn",
        "--config", "r/python.lang.best-practice.pass-body.pass-body-range",
        "--config", "r/python.lang.best-practice.pdb.python-debugger-found",
        "--config", "r/python.lang.best-practice.sleep.arbitrary-sleep",
        "--config", "r/python.lang.best-practice.unspecified-open-encoding.unspecified-open-encoding",
        "--config", "r/python.lang.correctness.baseclass-attribute-override.baseclass-attribute-override",
        "--config", "r/python.lang.correctness.cannot-cache-generators.cannot-cache-generators",
        "--config", "r/python.lang.correctness.common-mistakes.default-mutable-dict.default-mutable-dict",
        "--config", "r/python.lang.correctness.common-mistakes.default-mutable-list.default-mutable-list",
        "--config", "r/python.lang.correctness.common-mistakes.is-comparison-string.identical-is-comparison",
        "--config", "r/python.lang.correctness.common-mistakes.is-comparison-string.string-is-comparison",
        "--config", "r/python.lang.correctness.common-mistakes.is-not-is-not.is-not-is-not",
        "--config", "r/python.lang.correctness.common-mistakes.string-concat-in-list.string-concat-in-list",
        "--config", "r/python.lang.correctness.concurrent.uncaught-executor-exceptions",
        "--config", "r/python.lang.correctness.dict-modify-iterating.dict-del-while-iterate",
        "--config", "r/python.lang.correctness.exceptions.exceptions.raise-not-base-exception",
        "--config", "r/python.lang.correctness.exit.use-sys-exit",
        "--config", "r/python.lang.correctness.file-object-redefined-before-close.file-object-redefined-before-close",
        "--config", "r/python.lang.correctness.list-modify-iterating.list-modify-while-iterate",
        "--config", "r/python.lang.correctness.pdb.pdb-remove",
        "--config", "r/python.lang.correctness.pytest-assert_match-after-path-patch.pytest-assert_match-after-path-patch",
        "--config", "r/python.lang.correctness.return-in-init.return-in-init",
        "--config", "r/python.lang.correctness.return-in-init.yield-in-init",
        "--config", "r/python.lang.correctness.sync-sleep-in-async-code.sync-sleep-in-async-code",
        "--config", "r/python.lang.correctness.tempfile.flush.tempfile-without-flush",
        "--config", "r/python.lang.correctness.tempfile.mktemp.tempfile-insecure",
        "--config", "r/python.lang.correctness.test-is-missing-assert.test-is-missing-assert",
        "--config", "r/python.lang.correctness.unchecked-returns.unchecked-subprocess-call",
        "--config", "r/python.lang.correctness.useless-comparison.no-strings-as-booleans",
        "--config", "r/python.lang.correctness.useless-eqeq.useless-eqeq",
        "--config", "r/python.lang.correctness.writing-to-file-in-read-mode.writing-to-file-in-read-mode",
        "--config", "r/python.lang.maintainability.improper-list-concat.improper-list-concat",
        "--config", "r/python.lang.maintainability.is-function-without-parentheses.is-function-without-parentheses",
        "--config", "r/python.lang.maintainability.return.code-after-unconditional-return",
        "--config", "r/python.lang.maintainability.return.return-not-in-function",
        "--config", "r/python.lang.maintainability.useless-assign-keyed.useless-assignment-keyed",
        "--config", "r/python.lang.maintainability.useless-ifelse.useless-if-body",
        "--config", "r/python.lang.maintainability.useless-ifelse.useless-if-conditional",
        "--config", "r/python.lang.maintainability.useless-innerfunction.useless-inner-function",
        "--config", "r/python.lang.maintainability.useless-literal-set.useless-literal-set",
        "--config", "r/python.lang.maintainability.useless-literal.useless-literal",
        "--config", "r/python.requests.best-practice.use-raise-for-status.use-raise-for-status",
        "--config", "r/python.requests.best-practice.use-request-json-shortcut.python.requests.best-practice.use-request-json-shortcut",
        "--config", "r/python.requests.best-practice.use-response-json-shortcut.python.requests.best-practice.use-response-json-shortcut",
        "--config", "r/python.requests.best-practice.use-timeout.use-timeout",
        "--config", "r/python.sqlalchemy.correctness.bad-operator-in-filter.bad-operator-in-filter",
        "--config", "r/python.sqlalchemy.correctness.delete-where.delete-where-no-execute",
        "--config", "r/python.sqlalchemy.performance.performance-improvements.batch-import",
        "--config", "r/python.sqlalchemy.performance.performance-improvements.len-all-count",
        "--config", "r/trailofbits.python.numpy-in-pytorch-modules.numpy-in-pytorch-modules",
        "--config", "r/trailofbits.python.pytorch-tensor.pytorch-tensor"
    ]
    
    subprocess.run(semgrep_command, check=True)
    
    end_time = time.time()
    run_semgrep_time = end_time - start_time
    return run_semgrep_time

def delete_python_files(output_prefix, batch_number, num_files, lines_per_file):
    start_time = time.time()

    for i in range(num_files):
        line_number = batch_number * num_files * lines_per_file + i * lines_per_file + 1
        file_path = f"{output_prefix}_{line_number}.py"
        if os.path.exists(file_path):
            os.remove(file_path)
        #     print(f"Deleted {file_path}")
        # else:
        #     print(f"File not found: {file_path}")


    end_time = time.time()
    delete_files_time = end_time - start_time
    return delete_files_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON file and run Semgrep analysis.')
    parser.add_argument('json_file', type=str, help='The path to the JSON file.')

    args = parser.parse_args()

    json_filename = os.path.basename(args.json_file)
    output_prefix = os.path.splitext(json_filename)[0]

    start_time = time.time()

    split_json_time, split_times, semgrep_times, delete_times = split_json_to_python_files(args.json_file, output_prefix)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    print("\nDetailed timings per batch:")
    for i, (split_time, semgrep_time, delete_time) in enumerate(zip(split_times, semgrep_times, delete_times), start=1):
        print(f"Batch {i}: Semgrep time: {semgrep_time:.2f} s")