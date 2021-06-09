from . import __version__
import logging
import click


log = logging.getLogger(__name__)


def _setup_logging(ctx, obj, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO
    )

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Verbose logging enabled")
        log.debug(
            "Running %(prog)s, version %(version)s" % {
                "prog": ctx.find_root().info_name,
                "version": __version__
            }
        )

    return verbose


@click.group("contexts", invoke_without_command=True)
@click.option(
    "--verbose",
    is_flag=True,
    callback=_setup_logging,
    expose_value=False,
    is_eager=True,
    help="Enable DEBUG logging level")
@click.version_option(version=__version__)
@click.pass_context
def contexts(ctx):
    """ Attack Tree CLI
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help(), color=ctx.color)
        ctx.exit()


@contexts.command("tree-2-svg")
@click.argument("infile")
@click.argument("outfile")
def render_attack_tree_to_svg(infile, outfile):
    from .parser import parse_file
    import os
    import subprocess

    t = parse_file(infile)
    t.render_all_to_svg(out_path=outfile)


@contexts.command("bands")
def display_bands():
    from .parser import likelihood_metrics, Likelihood

    for l in Likelihood:
        m = likelihood_metrics[l]
        print("Likelihood({}): Lower={}, Center={}, Upper={}, Colour={}".format(
            l.value, m.lower, m.center, m.upper, m.colour
        ))


@contexts.command("legend")
@click.argument("outfile")
def render_legend(outfile):
    from .parser import get_legend_dot
    import os
    import subprocess

    t = get_legend_dot()
    dot_path = "legend.dot"
    try:
        with open(dot_path, "w") as f:
            f.write(t)
        subprocess.run("dot -Tsvg '{}' -o '{}'".format(dot_path, outfile), shell=True, check=True)
    finally:
        if os.path.exists(dot_path):
            os.remove(dot_path)


@contexts.command("batch")
@click.argument("attacktrees")
@click.argument("svgs")
def render_batch(attacktrees, svgs):
    from .parser import parse_file
    import os

    assert os.path.exists(attacktrees)
    assert os.path.exists(svgs)

    for fn in os.listdir(attacktrees):
        if fn.endswith(".txt"):
            infile = os.path.join(attacktrees, fn)

            t = parse_file(infile)
            outfile = os.path.join(svgs, fn[:-len(".txt")] + ".svg")
            t.render_all_to_svg(out_path=outfile)


def main() -> None:
    contexts(obj={})
