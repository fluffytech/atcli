from typing import NamedTuple, List, Dict, Optional, Set, Tuple
from enum import Enum, unique
import textwrap
import math
import logging
import os
import colorsys

log = logging.getLogger(__name__)


@unique
class NodeType(Enum):

    AND = "and"
    OR = "or"
    SEQ = "seq"
    REF = "ref"


@unique
class Likelihood(Enum):

    RemoteChance = "REMOTE CHANCE"
    HighlyUnlikely = "HIGHLY UNLIKELY"
    Unlikely = "UNLIKELY"
    RealisticPossibility = "REALISTIC POSSIBILITY"
    Likely = "LIKELY"
    HighlyLikely = "HIGHLY LIKELY"
    AlmostCertain = "ALMOST CERTAIN"


_def_likelihood_upper_thresholds = (
    0.075,
    0.225,
    0.375,
    0.525,
    0.775,
    0.925,
    1.0
)


class LikelihoodMetrics(NamedTuple):

    lower: float
    center: float
    upper: float
    colour: str


likelihood_metrics = dict()  # type: Dict[Likelihood, LikelihoodMetrics]


def _initialise_likelihood_bands() -> None:
    prev = 0.0

    # In HSV, 0 = Red, 120 = Green.
    hue_start = 0.0
    hue_end = 0.33

    for likelihood, band in zip(Likelihood, _def_likelihood_upper_thresholds):
        center = prev + ((band - prev) / 2.0)
        hue = hue_start + ((hue_end - hue_start) * center)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.4, 1.0)
        colour = "\"#%02x%02x%02x\"" % (round(r * 255), round(g * 255), round(b * 255))
        likelihood_metrics[likelihood] = LikelihoodMetrics(
            lower=prev,
            center=center,
            upper=band,
            colour=colour,
        )

        prev = band


_initialise_likelihood_bands()


def get_likelihood_from_probability(val: float) -> Likelihood:

    ret = Likelihood.AlmostCertain

    for likelihood, band in zip(Likelihood, _def_likelihood_upper_thresholds):
        if val < band:
            ret = likelihood
            break

    if log.isEnabledFor(logging.DEBUG):
        log.debug("probability {} = {}".format(val, ret.value))
    return ret


parent_node_back_colour = "\"#f0f0f0\""


def get_legend_dot() -> str:
    lines = list()
    lines.append("digraph a {")
    lines.append("bgcolor=transparent;")
    # lines.append("rankdir=LR;")
    lines.append("style=invis;")
    edge_colour = "\"#000000\""
    text_colour = "\"#000000\""

    for li in Likelihood:
        n = list()
        n.append(li.value.replace(" ", "_"))
        n.append(" [")
        n.append("color={}".format(edge_colour))
        n.append(", label=\"{}\\n({}%)\"".format(li.value.title(), int(likelihood_metrics[li].center * 100)))
        n.append(", style=\"filled, rounded\"")
        n.append(", fontcolor={}".format(text_colour))
        n.append(", shape=rectangle")
        n.append(", fillcolor={}".format(likelihood_metrics[li].colour))
        n.append("];")

        lines.append("".join(n))

    lines.append("}")
    return "\n".join(lines)


class Node(NamedTuple):

    display_name: str
    node_id: str
    type: NodeType
    line_num: int
    likelihood: Likelihood
    split: bool
    file: str
    xrefs: Tuple[str, ...]

    def render_node(
            self,
            likelihood: Likelihood,
            edge_colour: str,
            is_split: bool,
            text_colour: str,
            back_colour: str) -> str:

        extras = list()

        extras.append(likelihood.value.title())

        extras.extend(self.xrefs)
        if is_split:
            extras.append("(sub-graph hidden)")

        n = list()

        if extras:

            e = list()
            e.append(self.display_name)
            e.extend(extras)

            n.append(self.node_id)
            n.append(" [")
            n.append("color={}".format(edge_colour))
            n.append(", shape=Mrecord")
            if is_split:
                n.append(", style=\"filled, bold, dashed\"")
            else:
                n.append(", style=\"filled, bold\"")

            n.append(", label=\"{{{}}}\"".format("|".join(e)))
            n.append(", fontcolor={}".format(text_colour))
            n.append(", fillcolor={}".format(back_colour))
            n.append("];")

        else:
            n.append(self.node_id)
            n.append(" [")
            n.append("color={}".format(edge_colour))
            n.append(", label=\"{}\"".format(self.display_name))
            n.append(", style=\"filled, rounded, bold\"")
            n.append(", fontcolor={}".format(text_colour))
            n.append(", shape=rectangle")
            n.append(", fillcolor={}".format(back_colour))
            n.append("];")

        return "".join(n)


def link(child: 'TreeNode', parent: 'TreeNode') -> None:
    assert parent is not None
    assert child.node.node_id not in parent.children_lookup
    parent.children_lookup[child.node.node_id] = child
    parent.children.append(child)
    child.parent = parent


def replace(old: 'TreeNode', new: 'TreeNode') -> None:
    # Everything should have parents (root node covers all)
    assert old.parent
    assert new.parent
    assert old.node.type == NodeType.REF
    assert new.node.type != NodeType.REF

    parent = old.parent
    assert old.node.node_id in parent.children_lookup
    idx = parent.children.index(old)
    assert idx >= 0
    parent.children.remove(old)
    parent.children.insert(idx, new)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Replacing reference {} on line {}".format(old.node.node_id, old.node.line_num))


def recurse_render_node(
        state: 'SplitState',
        node: 'TreeNode',
        done: Set[str],
        rendered_nodes: Set[str],
        node_defs: List[str],
        edge_defs: List[str],
        probability_map: Dict[str, float]) -> None:

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Recurse rendering of {}".format(node.node.node_id))

    if node.node.node_id in done:
        return

    probability = probability_map[node.node.node_id]
    likelihood = get_likelihood_from_probability(probability)
    back_colour = likelihood_metrics[likelihood].colour
    edge_colour = "\"#000000\""
    text_colour = "\"#000000\""
    gate_back_colour = "\"#000000\""
    gate_edge_colour = "\"#000000\""
    gate_text_colour = "\"#ffffff\""

    is_split = state.rendering and node.node.split

    # Start rendering if this is a target...
    if node.node.node_id == state.target:
        state.rendering = True

    if state.rendering:
        node_defs.append(node.node.render_node(
            likelihood=likelihood,
            edge_colour=edge_colour,
            is_split=is_split,
            back_colour=back_colour,
            text_colour=text_colour,
        ))
        rendered_nodes.add(node.node.node_id)

    done.add(node.node.node_id)

    if is_split:
        state.rendering = False

    if node.children:
        for child in node.children:
            recurse_render_node(
                state=state,
                node=child,
                done=done,
                node_defs=node_defs,
                edge_defs=edge_defs,
                rendered_nodes=rendered_nodes,
                probability_map=probability_map
            )

        parent = node.node.node_id

        if len(node.children) > 1:
            # Need to add a gate...
            parent = parent + "__{}".format(node.node.type.value.upper())
            if node.node.type == NodeType.SEQ:
                # We render sequences differently...
                if state.rendering:
                    seq = list()
                    seq.append(parent)
                    seq.append(" [")
                    seq.append("color={}".format(gate_edge_colour))
                    seq.append(", shape=Mrecord")
                    seq.append(", style=\"filled\"")
                    seq.append(", fillcolor={}".format(gate_back_colour))
                    seq.append(", fontcolor={}".format(gate_text_colour))
                    seq.append(", label=\"{{{}|{{".format("SEQ"))
                    for i in range(len(node.children)):
                        if i > 0:
                            seq.append("|")
                        seq.append("<{}>{}".format(i+1, i+1))

                    seq.append("}}\"];")
                    node_defs.append("".join(seq))
            else:
                if state.rendering:
                    gate = list()
                    gate.append(parent)
                    gate.append(" [")
                    gate.append("color={}".format(gate_edge_colour))
                    gate.append(", label=\"{}\"".format(node.node.type.value.upper()))
                    gate.append(", style=\"filled, rounded\"")
                    gate.append(", fillcolor={}".format(gate_back_colour))
                    gate.append(", fontcolor={}".format(gate_text_colour))
                    gate.append(", shape=rectangle")
                    gate.append("];")
                    node_defs.append("".join(gate))

            # Add an edge to the new logic gate
            if state.rendering:
                edge = list()
                edge.append("{} -> {}".format(node.node.node_id, parent))
                edge.append(" [style=solid, color={}];".format(edge_colour))
                edge_defs.append("".join(edge))

                if node.node.type == NodeType.SEQ:
                    for idx, child in enumerate(node.children):
                        if child.node.node_id in rendered_nodes:
                            edge = list()
                            edge.append("{}:{} -> {}".format(parent, idx+1, child.node.node_id))
                            edge.append(" [style=solid, color={}];".format(edge_colour))
                            edge_defs.append("".join(edge))
                else:
                    for child in node.children:
                        if child.node.node_id in rendered_nodes:
                            edge = list()
                            edge.append("{} -> {}".format(parent, child.node.node_id))
                            edge.append(" [style=solid, color={}];".format(edge_colour))
                            edge_defs.append("".join(edge))
        else:
            if state.rendering:
                for child in node.children:
                    if child.node.node_id in rendered_nodes:
                        edge = list()
                        edge.append("{} -> {}".format(parent, child.node.node_id))
                        edge.append(" [style=solid, color={}];".format(edge_colour))
                        edge_defs.append("".join(edge))

    # Restore rendering if on splie
    if is_split:
        state.rendering = True

    # Stop rendering if this is the target...
    if node.node.node_id == state.target:
        state.rendering = False


class Results(NamedTuple):
    dot: str
    splits: Tuple[str, ...]


class SplitState:

    def __init__(self, target: Optional[str]) -> None:
        self.target = target
        self.rendering = True if self.target is None else False


class Renderer(NamedTuple):

    lr: bool
    splits: Tuple[str, ...]
    probability_map: Dict[str, float]
    root: 'TreeNode'

    def _render_target_to_svg(self, base_path: str, target: str) -> None:
        import subprocess
        import os

        file_target = target
        if file_target and file_target.startswith("n_"):
            file_target = file_target[1:]

        svg_path = "{}{}.svg".format(base_path, file_target)
        log.info("Rendering to {}".format(svg_path))
        dot_path = "{}{}.dot".format(base_path, file_target)
        try:
            with open(dot_path, "w") as f:
                f.write(self.render(target=target if target else None))
            subprocess.run("dot -Tsvg '{}' -o '{}'".format(dot_path, svg_path), shell=True, check=True)
        finally:
            if os.path.exists(dot_path):
                os.remove(dot_path)

    def render_all_to_svg(self, out_path: str) -> None:
        assert out_path.endswith(".svg")
        base_path = out_path[:-len(".svg")]
        self._render_target_to_svg(base_path=base_path, target="")
        for split in self.splits:
            self._render_target_to_svg(base_path=base_path, target=split)

    def render(self, target: Optional[str]=None) -> str:

        state = SplitState(target=target)

        done = set()
        node_defs = list()
        edge_defs = list()
        rendered_nodes = set()

        recurse_render_node(
            node=self.root,
            done=done,
            node_defs=node_defs,
            edge_defs=edge_defs,
            probability_map=self.probability_map,
            state=state,
            rendered_nodes=rendered_nodes,
        )

        if not node_defs:
            raise RuntimeError("This graph contains no nodes!")

        lines = list()
        lines.append("digraph a {")
        lines.append("bgcolor=transparent;")
        if self.lr:
            lines.append("rankdir=LR;")
        lines.append("style=invis;")

        lines.extend(node_defs)
        lines.extend(edge_defs)
        lines.append("}")

        return "\n".join(lines)


class Working:

    def __init__(self) -> None:
        self.references = list()  # type: List[TreeNode]
        self.instances = dict()  # type: Dict[str, TreeNode]
        self.root = TreeNode(Node(
            node_id="ROOT",
            display_name="ROOT",
            type=NodeType.OR,
            line_num=-1,
            likelihood=Likelihood.Likely,
            split=False,
            file="",
            xrefs=tuple(),
        ))
        self.probability_map = dict()
        self.lr = False
        self.splits = list()

        # We keep a record of files processed (we process the first differently)
        self.file_count = 0

    def parse_file(self, fn: str) -> None:

        with open(fn, "r") as f:
            self.file_count += 1

            prev_prefixes = list()
            parents = list()
            current_prefix = 0
            line_num = 0
            node = None
            parent = None

            for ln in f:
                line_num += 1

                if self.file_count == 1 and line_num == 1:
                    if ln.startswith("#") and ln[1:].strip().lower() == "lr":
                        self.lr = True

                if not ln.strip().startswith("#"):
                    if ln.strip():
                        # Remove trailing spaces
                        ln = ln.rstrip()
                        new_prefix = len(ln) - len(ln.lstrip())
                        if new_prefix > current_prefix:
                            # Move right...
                            prev_prefixes.append(current_prefix)
                            parents.append(parent)
                            parent = node
                            node = None
                        elif new_prefix < current_prefix:
                            # Move left...
                            while new_prefix < current_prefix:
                                current_prefix = prev_prefixes.pop()
                                node = parent
                                parent = parents.pop()
                                if current_prefix == new_prefix:
                                    break
                                elif current_prefix < new_prefix:
                                    raise RuntimeError("error!")
                            if not new_prefix:
                                assert len(prev_prefixes) == 0
                        current_prefix = new_prefix
                        if ln.startswith("include::"):
                            relative_path = ln[len("include::"):]
                            dirname = os.path.dirname(fn)
                            included_path = os.path.join(dirname, relative_path)
                            if not os.path.isfile(included_path):
                                raise RuntimeError("Cannot find included file: {}".format(included_path))
                            log.info("Including file: {}".format(included_path))
                            self.parse_file(included_path)
                        else:
                            node = self.add_node(
                                node=parse_node(ln[new_prefix:], line_num=line_num, file=fn),
                                parent=parent)
                            log.debug("Depth({}): {}".format(len(prev_prefixes), node))

    def _recurse_analyse(self, node: 'TreeNode', started_nodes: Set[str], probability_map: Dict[str, float]) -> float:
        try:
            return probability_map[node.node.node_id]
        except KeyError:
            pass

        if node.node.node_id in started_nodes:
            log.info("cyclic detected!")
            return likelihood_metrics[Likelihood.AlmostCertain].center
        started_nodes.add(node.node.node_id)
        if node.children:
            if node.node.type == NodeType.OR:
                probability = 0.0
                for child in node.children:
                    probability = max(probability, self._recurse_analyse(
                        node=child, started_nodes=started_nodes, probability_map=probability_map))
            else:
                log.debug("combine gate - {}".format(node.node.node_id))
                probability = 1.0
                for child in node.children:
                    p = self._recurse_analyse(
                        node=child, started_nodes=started_nodes, probability_map=probability_map)
                    log.debug("child {}.{} = {}".format(node.node.node_id, child.node.node_id, p))
                    probability *= p
        else:
            probability = likelihood_metrics[node.node.likelihood].center
        log.debug("setting probability for {} = {}".format(node.node.node_id, probability))
        probability_map[node.node.node_id] = probability
        return probability

    def calculate_path_probability(self) -> None:
        started_nodes = set()
        probability_map = dict()

        # No point analysing if there are no nodes!
        if self.root.children:
            self._recurse_analyse(
                node=self.root.children[0],
                started_nodes=started_nodes,
                probability_map=probability_map
            )
        self.probability_map = probability_map

    def resolve_references(self) -> None:
        for ref in self.references:
            try:
                i = self.instances[ref.node.node_id]
            except KeyError:
                raise RuntimeError("Unresolved reference on line {}".format(ref.node.line_num)) from None
            else:
                replace(old=ref, new=i)

    def add_node(self, node: Node, parent: Optional['TreeNode']) -> 'TreeNode':
        parent = parent or self.root
        if node.type == NodeType.REF:
            tn = TreeNode(node=node)
            link(child=tn, parent=parent)

            # Can have as many of these as we like
            self.references.append(tn)

        else:
            assert node.type in (NodeType.OR, NodeType.AND, NodeType.SEQ)
            # We cannot duplicate this instance, so throw an error if already exists
            try:
                n = self.instances[node.node_id]
            except KeyError:
                pass
            else:
                raise RuntimeError("Duplicate node name on line {}".format(n.node.line_num))

            tn = TreeNode(node=node)
            link(child=tn, parent=parent)
            self.instances[tn.node.node_id] = tn
            if tn.node.split:
                self.splits.append(tn.node.node_id)

        return tn


class TreeNode:

    def __init__(self, node: Node) -> None:
        self.node = node
        self.children = list()  # type: List[TreeNode]
        self.children_lookup = dict()  # type: Dict[str, TreeNode]
        self.parent = None  # type: Optional[TreeNode]

    def __repr__(self) -> str:
        return "Node: {}, Parent: {}".format(self.node.node_id, self.parent)


ALLOWED_CHARS = set("1234567890abcdefghijklmnopqrstuvwxyz_.-? (),")
ALLOWED_XREF_CHARS = set("1234567890abcdefghijklmnopqrstuvwxyz_.-? (),[]")
ALLOWED_NODE_ID_CHARS = set("1234567890abcdefghijklmnopqrstuvwxyz_")


def wrap_string(val: str) -> str:
    wrap_length = max(20, int(math.sqrt(len(val)) + 0.6) * 2)
    return "\\n".join(textwrap.wrap(val, width=wrap_length))


def parse_node(val: str, line_num: int, file: str) -> Node:
    val = val.strip()
    if not val:
        raise RuntimeError("Empty node!")
    node_type = NodeType.OR
    likelihood = Likelihood.AlmostCertain
    node_split = False
    xrefs = list()

    if val[-1] == ")":
        # Reverse scan for a trailer...
        pos = val.rfind("(", 0, -1)
        if pos > 0:
            trailer = val[pos+1:-1]
            val = val[:pos].strip()
            for x in [x.strip() for x in trailer.split(",")]:
                x_orig = x

                x = x.upper()

                if x == "AND":
                    node_type = NodeType.AND
                elif x == "OR":
                    node_type = NodeType.OR
                elif x == "SEQ":
                    node_type = NodeType.SEQ
                elif x == "REF":
                    node_type = NodeType.REF
                elif x == "SPLIT":
                    node_split = True
                elif x.startswith("X"):
                    xref = x_orig[1:].strip()
                    if xref:
                        bad_chars = set(xref.lower()).difference(ALLOWED_XREF_CHARS)
                        if bad_chars:
                            raise RuntimeError(
                                "Bad extra info chars ('{}') at line {}".format("".join(sorted(bad_chars)), line_num))

                        xrefs.append(wrap_string(xref))
                else:
                    try:
                        likelihood = Likelihood(x)
                    except ValueError:
                        raise RuntimeError("Unknown trailer variable ({}) at line {}".format(x, line_num)) from None

    bad_chars = set(val.lower()).difference(ALLOWED_CHARS)
    if bad_chars:
        raise RuntimeError("Bad chars ('{}') at line {}".format("".join(sorted(bad_chars)), line_num))

    node_id = "n_" + val.lower()
    bad_node_chars = set(node_id).difference(ALLOWED_NODE_ID_CHARS)
    for c in bad_node_chars:
        node_id = node_id.replace(c, "_")

    return Node(
        node_id=node_id,
        display_name=wrap_string(val),
        type=node_type,
        line_num=line_num,
        likelihood=likelihood,
        split=node_split,
        file=file,
        xrefs=tuple(xrefs),
    )


def parse_file(fn: str) -> Renderer:
    fn = os.path.abspath(os.path.expanduser(fn))

    w = Working()

    w.parse_file(fn)

    w.resolve_references()
    w.calculate_path_probability()

    if not w.root.children:
        raise RuntimeError("Empty graph!")

    return Renderer(
        lr=w.lr,
        probability_map=w.probability_map,
        root=w.root.children[0],
        splits=tuple(w.splits),
    )




