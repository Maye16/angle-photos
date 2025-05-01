import numpy as np
from correctedInterfaceRadiusFinder import analyze_meniscus

width = 100*10**-6

image_path = "Figures/3.png"
results = analyze_meniscus(image_path, enable_plot=False)

alpha = results["contact_angle"]
beta = results["beta"]
radius = results["radius"]

theta = 180/np.pi * np.arccos(width/2*radius)
print(theta)


