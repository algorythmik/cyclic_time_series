import subprocess
import time
import getpass
import git


def sync(source, destination, encrypted=False, quiet=False, additional=''):
    """Helper function for syncing from s3."""
    sync_cmd = [
        'aws',
        's3',
        'sync',
        source,
        destination
    ]
    if additional:
        sync_cmd.extend(additional.split())

    if quiet:
        sync_cmd.append('--quiet')

    subprocess.run(sync_cmd, check=True)


def make_signature(sha, run_name=None):
    """
    Creates a signature used to identify a run of an experiment.
    Signatures are unix timestamp, sha, and username
    """
    sha_length = 12
    sha = sha[:sha_length]
    signature = time.strftime('%Y%m%d_%H%M%S')
    user = getpass.getuser().lower()
    signature = '_'.join([signature, sha, user])
    if run_name:
        signature += f'_{run_name}'

    return signature


def repo_is_clean():
    repo = git.Repo(search_parent_directories=True)
    assert repo.untracked_files == [],\
        "There are untracked files in your repo. Please stage them!"
    assert repo.is_dirty() is not True,\
        ("Please make sure that you have"
         "a clean git environment, commit your changes!")
    branch = repo.active_branch.name
    commits_ahead = len(list(repo.iter_commits(f'{branch}@{{u}}..{branch}')))
    assert commits_ahead == 0, ("Please make sure you have pushed to the"
                                " upstream branch!")


def upload_run(run, s3_bucket, project_name):
    s3_bucket = '/'.join([s3_bucket, project_name, 'models'])
    sha = git.Repo(search_parent_directories=True).head.object.hexsha
    signature = make_signature(sha)
    source = run.info.artifact_uri[8:-len('artifacts')]
    destination = '/'.join([s3_bucket, signature]) + '/'
    sync(source, destination)
