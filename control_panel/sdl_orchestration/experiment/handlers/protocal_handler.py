from typing import Optional

from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.experiment.handlers.reagents_handler import ReagentMappingHandler
from sdl_orchestration.experiment.samples.plate96well import Plate96Well

SYNTHESIS_TARGET = """\n####################\n# REPLACE ME WITH THE REST OF THE TARGETS\n####################\n"""


class ProtocolFactory:
    """
    This class is responsible for creating the protocol.

    All supported protocols are defined in the protocol_map.
    """

    SYNTHESIS_TARGET_PLACE_HOLDER = SYNTHESIS_TARGET

    def __init__(self):
        self.protocol_map = {
            "lipid_synthesis_template": self._synthesis,
        }

    def _synthesis(self, *args, **kwargs):
        assert "targets" in kwargs, "Targets not found."
        assert isinstance(
            kwargs["targets"], Plate96Well
        ), "Targets must be a Plate96Well object."
        assert "parse_source" in kwargs, "Parse source not found."
        assert "parse_dest" in kwargs, "Parse destination not found."
        targets = kwargs["targets"]
        parse_source = kwargs["parse_source"]
        parse_dest = kwargs["parse_dest"]

        self._parse_synthesis_protocol(
            targets,
            parse_source,
            parse_dest,
        )

    def _parse_synthesis_protocol(
        self,
        targets: Plate96Well,
        parse_source: str,
        parse_dest: str,
        experiment_id: Optional[ObjectId] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        This method parses the synthesis protocol file's content.

        This method currently applies the parsing on a given set of
        protocol, including:
            - "lipid_synthesis": this computes the required volume of each
            target lipid to be added to the lipid mixture. This will create
            a specialized protocol for the Opentron. We automatically
            save the protocol to the destination path.

        Args:
            targets (Plate96Well): The target plate.
            parse_source (str): The path to the source protocol file.
            parse_dest (str): The path to the destination protocol file.
            experiment_id (Optional[ObjectId], optional): The id of the
        """

        # We first resolve the reagent mapping for this target
        reagent_handler = ReagentMappingHandler(targets, experiment_id)
        mapping_string = reagent_handler.resolve_reverse_mapping()

        logger.info(f"Mapping string: {mapping_string}")

        # We then parse the protocol file
        with open(parse_source, "r") as f:
            content = f.read()
            content = content.replace(
                self.SYNTHESIS_TARGET_PLACE_HOLDER, mapping_string
            )

        # We then write the parsed content to the destination path
        with open(parse_dest, "w") as f:
            f.write(content)

        logger.info(f"Protocol file written to {parse_dest}")

    def get_protocal(self, protocol_name, *args, **kwargs):
        if protocol_name not in self.protocol_map:
            raise ValueError(f"Protocol {protocol_name} not found.")
        return self.protocol_map[protocol_name](*args, **kwargs)
