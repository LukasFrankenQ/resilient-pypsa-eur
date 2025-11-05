##### Summarises the changes to the default PyPSA-Eur that should be tracked to be noted in the paper

- cutout for 2024, electricity demand is derived using that cutout in https://github.com/L-vdM/EU-renewable-energy-modelling-framework
- added decentral/rural biomass boilers
- added must-run to lignite and hard coal without considering DH
- fixed industrial gas demand to actual
- made heating p_nom_extendable = False
- fixed p_min/max_pu of decentral heating to ensure realistic operation
- removed fixed carbon constraints and instead added (UK) ETS
- set basically everything non-extendable
- 