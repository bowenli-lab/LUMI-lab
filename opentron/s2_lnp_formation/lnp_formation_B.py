from opentrons import protocol_api

# 1. take the (1)lipid well out from shaker into opentron 2, assuming (2)Et maste rmix in cold box in one of the opentron 2 slot.
# 2. take 5 ul master mix into the new well
# 3. take the 5ul lipid from the lipid well into (3)a new well,
# 4. take 30 ul (4)Aqueous Phase Total (mRNA) into the new well, “mix after” 3 times

# metadata
metadata = {
    "protocolName": "Step 2 LNP Formation",
    "author": "Haotian Cui <subercui@gmai.com>",
    "description": "Form LNP by combining lipid, master mix, and mRNA.",
}

# requirements
requirements = {"robotType": "OT-2", "apiLevel": "2.18"}

# production confirmed
volume_et_master_mix = 2.6
volume_lipid = 4
volume_mrna = 19.7


# protocol run function
def run(protocol: protocol_api.ProtocolContext):
    # DECLARE THE EQUIPMENT INFO AND STATUS
    # we will need 4 different wells, 1. new lnp plate, 2. lipid plate,
    # 3. ethanol master mix plate, 4. Aqueous mrna plate

    # labware
    tipracks = [
        protocol.load_labware("opentrons_96_tiprack_20ul", str(slot))
        for slot in [4, 5, 7]
    ]

    lipid_plate = protocol.load_labware("servicebio_96_wellplate_200ul", location="1")
    new_lnp_plate = protocol.load_labware("servicebio_96_wellplate_200ul", location="2")

    cold_deepwell = protocol.load_labware(
        "customizedcoolerplate_96_wellplate_1300ul", location="6"
    )

    # pipettes
    left_multi_pipette = protocol.load_instrument(
        "p20_multi_gen2", mount="left", tip_racks=tipracks
    )

    # ACTION COMMANDS
    # take 5 ul master mix into the new well
    left_multi_pipette.transfer(
        volume_et_master_mix,
        cold_deepwell.columns_by_name()["2"],  # experiment B special
        new_lnp_plate.wells(),
        new_tip="once",
        trash=False,
    )

    lipid_wells_bottom = [well.bottom(-2.8) for well in lipid_plate.wells()]
    new_lnp_wells_bottom = [well.bottom(-2.8) for well in new_lnp_plate.wells()]

    # take the 5ul lipid from the lipid well into a new well
    left_multi_pipette.transfer(
        volume_lipid,
        lipid_wells_bottom,
        new_lnp_wells_bottom,
        new_tip="always",
        trash=False,
        mix_before=(1, 10),
        blow_out=True,
        blowout_location="destination well",
    )

    # Transfer MC3 and SM102 control to the
    # LNP plate using first two tips and not touch the LNP plate
    new_lnp_plate.set_offset(x=0, y=0, z=5)
    left_multi_pipette.transfer(
        volume_lipid,
        cold_deepwell.columns_by_name()["7"],  # experiment B special
        new_lnp_plate.columns_by_name()["12"],
        new_tip="always",
        trash=False,
    )
    cold_deepwell.set_offset(x=0, y=0, z=0)

    # take 30 ul mRNA into the new well
    left_multi_pipette.transfer(
        volume_mrna,
        cold_deepwell.columns_by_name()["3"],
        new_lnp_wells_bottom,
        mix_after=(6, 20),
        new_tip="always",
        trash=False,
    )
