"""
Submit a job to qsub in the Tanay Group lab.
"""

from argparse import ArgumentParser
from binascii import crc32
from datetime import datetime
from glob import glob
from tgutils.application import lock_file
from time import sleep
from typing import List
from typing import Optional
from typing import Tuple

import os
import shlex
import stat
import subprocess
import sys


class Qsubber:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Submit a job to qsub in the Tanay Group lab.
    """

    def __init__(self) -> None:
        parser = ArgumentParser(description="Submit a job to qsub in the Tanay Group lab.")

        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Whether to emit verbose messages.')

        parser.add_argument('-j', '--job-id', metavar='STR', default='?',
                            help='A job identifier for better messages.')

        parser.add_argument('-s', '--slots', metavar='SLOTS', default=None,
                            help='The number of slots to use.')

        parser.add_argument('-S', '--size', metavar='SIZE', default=None,
                            help='The number of related jobs that are currently being submitted; '
                            'if specified, and --slots is not, then the number of slots is '
                            'computed automatically for optimal usage of the cluster.')

        parser.add_argument('-I', '--index', metavar='INDEX', default=None,
                            help='The index of the job in the related jobs currently being '
                            'submitted, for maximizing the distribution of jobs across hosts.')

        parser.add_argument('-m', '--min-slots', metavar='MIN', default='1',
                            help='The minimal number of slots to assign to each job.')

        parser.add_argument('-M', '--max-slots', metavar='MAX', default=None,
                            help='The maximal number of slots to assign to each job.')

        parser.add_argument('-e', '--every', metavar='SECONDS', default='15',
                            help='Qsub actually schedules jobs every this number of seconds '
                                 '(default: 15).')

        parser.add_argument('-x', '--extra', metavar='STR', default='',
                            help='Additional low-level flags to pass to qsub (default: None).')

        parser.add_argument('-o', '--offset', metavar='SECONDS', default='3',
                            help='Qsub is using this offset from round times to schedule jobs '
                                 '(default: 3).')

        default_tmp_dir = os.getenv('QSUB_TMP_DIR', '.qsub')
        parser.add_argument('-t', '--tmp_dir', metavar='PATH', default=default_tmp_dir,
                            help='A unique job identifier (default: $QSUB_TMP_DIR, %s).'
                            % default_tmp_dir)

        parser.add_argument('command', metavar='COMMAND',
                            help='The shell command to execute.')

        parser.add_argument('arguments', metavar='ARGUMENT', nargs='*',
                            help='The shell command arguments.')

        args = parser.parse_args()

        self.verbose = bool(args.verbose)
        self.job_id = str(args.job_id or '_')
        self.slots = str(args.slots or _default_slots(args.index, args.size,
                                                      args.min_slots, args.max_slots))
        self.tmp_dir = os.path.abspath(args.tmp_dir)
        self.every = int(args.every)
        self.offset = int(args.offset)
        self.command_arguments = [args.command] + args.arguments
        self.extra = args.extra or ''

        os.makedirs(self.tmp_dir, exist_ok=True)
        self.lock_path = os.path.join(self.tmp_dir, 'lock')
        self.lock_fd = os.open(self.lock_path, os.O_CREAT)

    def run(self) -> int:
        """
        Run the submitted job using the command line options.
        """
        with lock_file(self.lock_path, self.lock_fd):
            deadline, array_path_prefix = self._identify_array()
            is_master, job_path_prefix = self._create_job_in_array(array_path_prefix)

        _wait_for_almost_deadline(deadline)

        with lock_file(self.lock_path, self.lock_fd):
            Qsubber._close_array(array_path_prefix)

        if is_master:
            self._submit_array(array_path_prefix)

        job_status = Qsubber._wait_for_job_completion(job_path_prefix)

        with lock_file(self.lock_path, self.lock_fd):
            self._print_job_results(job_status, job_path_prefix)
            Qsubber._remove_job_from_array(array_path_prefix)

        if job_status == 0:
            _remove_files(job_path_prefix)

        return job_status

    def _create_job_in_array(self, array_path_prefix: str) -> Tuple[bool, str]:
        array_size_path = array_path_prefix + '.size'
        if os.path.exists(array_size_path):
            is_master = False
            with open(array_size_path, 'r') as array_size_file:
                array_size = int(array_size_file.read())
        else:
            is_master = True
            array_size = 0

        array_size += 1
        job_index_in_array = array_size

        job_path_prefix = array_path_prefix.replace('.array', '.i%s' % job_index_in_array)
        self._write_job_run_file(job_path_prefix)

        if is_master:
            Qsubber._write_array_run_file(array_path_prefix)
            self._write_array_submit_file(array_path_prefix)

        with open(array_size_path, 'w') as array_size_file:
            array_size_file.write('%s\n' % array_size)

        if self.verbose:
            _print('%s - tg_qsub - INFO - Submitted job: %s in: %s.*'
                   % (datetime.now(), self.job_id, job_path_prefix))

        return is_master, job_path_prefix

    @staticmethod
    def _remove_job_from_array(array_path_prefix: str) -> None:
        array_size_path = array_path_prefix + '.size'
        with open(array_size_path, 'r') as array_size_file:
            array_size = int(array_size_file.read())
            assert array_size > 0

        array_size -= 1
        if array_size == 0:
            _remove_files(array_path_prefix)
            return

        with open(array_size_path, 'w') as array_size_file:
            array_size_file.write('%s\n' % array_size)

    def _identify_array(self) -> Tuple[int, str]:
        now = int(datetime.now().timestamp())
        array_id = (now - self.offset) // self.every
        deadline = (array_id + 1) * self.every + self.offset

        time_left = deadline - now
        if time_left < 3:
            array_id += 1
            time_left += self.every
            deadline += self.every

        while True:
            array_name = 'a%s.s%s' % (array_id % 10000, self.slots)
            if self.extra:
                array_name += '.%x' % crc32(self.extra.encode('utf-8'))
            array_path_prefix = os.path.join(self.tmp_dir, array_name + '.array')
            if not os.path.exists(array_path_prefix + '.closed'):
                return deadline, array_path_prefix
            array_id += 1
            deadline += self.every

    def _write_job_run_file(self, job_path_prefix: str) -> None:
        job_run_path = job_path_prefix + '.run.sh'
        with open(job_run_path, 'w') as job_run_file:
            job_run_file.write('#!/bin/sh\n')
            if '-' not in self.slots:
                job_run_file.write("export DYNAMAKE_JOBS='%s'\n" % self.slots)
            job_run_file.write("echo `date +'%F %T.%N' | sed 's/......$//'` ")
            job_run_file.write('- tg_qsub - Running %s on `hostname` using %s slots\n'
                               % (job_run_path, self.slots))
            job_run_file.write(' '.join([shlex.quote(argument)
                                         for argument in self.command_arguments]))
            job_run_file.write('\n')
        _chmod_x(job_run_path)

    @staticmethod
    def _write_array_run_file(array_path_prefix: str) -> None:
        array_run_path = array_path_prefix + '.run.sh'
        job_path_prefix = array_path_prefix.replace('.array', '.i$SGE_TASK_ID')
        with open(array_run_path, 'w') as array_run_file:
            array_run_file.write('#!/bin/sh\n')
            array_run_file.write('source ~/.bashrc\n')
            array_run_file.write('cd %s\n' % os.path.abspath(os.getcwd()))
            array_run_file.write('if %s.run.sh > %s.output 2>&1\n'
                                 % (job_path_prefix, job_path_prefix))
            array_run_file.write('then\n')
            array_run_file.write('    touch %s.success\n' % job_path_prefix)
            array_run_file.write('else\n')
            array_run_file.write('    touch %s.failure\n' % job_path_prefix)
            array_run_file.write('fi\n')
        _chmod_x(array_run_path)

    def _write_array_submit_file(self, array_path_prefix: str) -> None:
        array_submit_path = array_path_prefix + '.submit.sh'
        job_path_prefix = array_path_prefix.replace('.array', '.i$TASK_ID')
        array_job_name = os.path.basename(array_path_prefix).replace('.array', '')
        with open(array_submit_path, 'w') as array_submit_file:
            array_submit_file.write('#!/bin/sh\n')
            array_submit_file.write('qsub ')
            if self.extra:
                array_submit_file.write('%s ' % self.extra)
            array_submit_file.write('-N %s ' % array_job_name)
            array_submit_file.write('-t 1-`cat %s.size` ' % array_path_prefix)
            array_submit_file.write('-j y ')
            array_submit_file.write('-b y ')
            array_submit_file.write("-o '%s.log' " % job_path_prefix)
            array_submit_file.write('-sync y ')
            if self.slots != '1':
                array_submit_file.write('-pe threads %s ' % self.slots)
            array_submit_file.write('%s.run.sh ' % array_path_prefix)
            array_submit_file.write('> %s.log 2>&1\n' % array_path_prefix)
        _chmod_x(array_submit_path)

    @staticmethod
    def _close_array(array_path_prefix: str) -> None:
        open(array_path_prefix + '.closed', 'w').close()

    def _submit_array(self, array_path_prefix: str) -> None:
        if self.verbose:
            _print('%s - tg_qsub - INFO - Submitted array: %s.*'
                   % (datetime.now(), array_path_prefix))
        subprocess.run(array_path_prefix + '.submit.sh', check=False)

    @staticmethod
    def _wait_for_job_completion(job_path_prefix: str) -> int:
        job_success_path = job_path_prefix + '.success'
        job_failure_path = job_path_prefix + '.failure'
        while True:
            if os.path.exists(job_success_path):
                return 0
            if os.path.exists(job_failure_path):
                return 1
            sleep(1)

    def _print_job_results(self, job_status: int, job_path_prefix: str) -> None:
        job_output_path = job_path_prefix + '.output'
        with open(job_output_path, 'r') as output_file:
            text = output_file.read()
            if text:
                _print(text)
        if job_status > 0 and self.verbose:
            _print('%s - tg_qsub - ERROR - Failed job: %s in: %s.*'
                   % (datetime.now(), self.job_id, job_path_prefix))


def _wait_for_almost_deadline(deadline: int) -> None:
    time_left = deadline - int(datetime.now().timestamp())
    if time_left > 2:
        sleep(time_left - 2)


def _remove_files(path_prefix: str) -> None:
    for path in glob(path_prefix + '.*'):
        os.remove(path)


def _default_slots(_index: Optional[str],  # pylint: disable=too-many-return-statements
                   size: Optional[str], min_slots: Optional[str], max_slots: Optional[str]) -> str:
    if size is None:
        # No size specified, use 1 slot.
        return '1'

    configuration = _configuration()
    hosts_count = len(configuration)
    processors_count = sum(configuration)

    int_size = int(size)
    if int_size <= hosts_count:
        # Less jobs than hosts, completely take over each used host.
        if min_slots is not None:
            if max_slots is not None:
                return '%s-%s' % (min_slots, max_slots)
            return '%s-' % min_slots
        if max_slots is not None:
            return max_slots
        return '1-'

    average_slots = processors_count / int_size
    use_slots = int(average_slots)
    if use_slots < 1:
        # Heavy contention, use one slot.
        return '1'

    # Several jobs per host, each using several processors.

    use_slots = _slots_per_job(configuration, int_size)

    if min_slots is not None:
        int_min_slots = int(min_slots)
        if use_slots < int_min_slots:
            use_slots = int_min_slots

    if max_slots is not None:
        int_max_slots = int(max_slots)
        if use_slots > int_max_slots:
            use_slots = int_max_slots

    # Low contention, allow maximal number of jobs while taking over all(most) processors.
    return str(use_slots)


def _configuration() -> List[int]:
    configuration: List[int] = []
    qhost = subprocess.run(['qhost', '-q'], stdout=subprocess.PIPE, check=True)
    for line in qhost.stdout.decode('utf-8').split('\n'):
        if line:
            fields = line.split()
            parts = fields[-1].split('/')
            if len(parts) == 3:
                configuration.append(int(parts[2]) - int(parts[1]))
    return configuration


def _slots_per_job(configuration: List[int], size: int) -> int:
    maximal_slots = 1
    candidate_slots = 1
    while True:
        candidate_slots += 1
        jobs = 0
        for processors in configuration:
            jobs += processors // candidate_slots
        if jobs < size:
            return maximal_slots
        maximal_slots = candidate_slots


def _chmod_x(path: str) -> None:
    stat_result = os.stat(path)
    os.chmod(path, stat_result.st_mode | stat.S_IEXEC)


def _print(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.write('\n')
    sys.stderr.flush()


def main() -> None:
    """
    Submit a job to qsub in the Tanay Group lab.
    """
    sys.exit(Qsubber().run())


if __name__ == '__main__':
    main()
