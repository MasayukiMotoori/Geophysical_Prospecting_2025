# Inversion of induced polarization parameters from time-domain electromagnetic data using the Debye Decomposition model 

**Masayuki Motoori¹², Lindsey J. Heagy¹, Gosuke Hoshino², Kunpei Nagase², Takumi Sato²**  
¹ University of British Columbia Geophysical Inversion Facility  
² Japan Organization for Metals and Energy Security (JOGMEC)

### Abstract

Time-domain electromagnetics (TEM) is sensitive to variations in resistivity and chargeability, which are diagnostic physical properties of seafloor hydrothermal deposits. Several surveys using the Waseda Integrated Seafloor Time-domain EM (WISTEM) have been conducted. Negative transients, which are attributed to induced polarization (IP) effects, have been observed in data collected over a known deposit in the Okinawa Trough (2018).

The Cole--Cole type model is a common parameterization of complex resistivity due to IP effects; however, it implicitly assumes a wide frequency band, which is not necessarily covered by the frequency band of the TEM response from a chargeable target. We show a synthetic study that TEM inversion using the Cole--Cole model suffers from significant instability, which we attribute to the discrepancy between the model’s assumed frequency band and the frequency band of the data.

The Debye Decomposition model provides a way to bridge this gap by allowing explicit selection of the relaxation time band over which the influence of chargeability in the data behaves distinctly from the influence of resistivity. In this paper, we introduce a workflow for inverting TEM data using a Debye Decomposition model. We demonstrate how to set the relaxation time band when inverting TEM data. We then compare the model recovered from the TEM data with a model recovered by inverting SIP data from the target. When inverting SIP data, we use the same Debye Decomposition model as was used for the TEM data. In this way, we tailor the frequency band to the relaxation time band, which we refer to as a reconciling process. Using a synthetic example, we demonstrate that the inversion results obtained from the TEM data are consistent with the reconciled true model obtained from the SIP response. 

We also present a field application and show that the method recovers reasonable and interpretable IP parameters for seafloor hydrothermal deposits.

Keywords: Electromagnetics, Inversion, Data processing, Induced Polarization, Marine
