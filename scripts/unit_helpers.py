def tj_to_twh(tj):
    """
    Convert energy from terajoules (TJ) to terawatt-hours (TWh).

    Parameters
    ----------
    tj : float or array-like
        Energy in terajoules.

    Returns
    -------
    float or array-like
        Equivalent energy in terawatt-hours.
    """
    # Conversion factor: 1 TJ = 1 / 3600 TWh
    return tj / 3600.0


def bcm_to_twh(bcm, twh_per_bcm=10.467):
    """
    Convert natural gas volume in bcm (10^9 m^3) to energy in TWh,
    assuming a fixed energy content per bcm.
    
    Parameters
    ----------
    bcm : float or array-like
        Volume of natural gas in billion cubic meters (m^3 × 10^9).
    twh_per_bcm : float
        Energy in TWh assumed per bcm. Default = 10.467 TWh / bcm.
    
    Returns
    -------
    float or array-like
        Equivalent energy in TWh.
    """
    return bcm * twh_per_bcm