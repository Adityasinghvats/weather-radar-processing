def calc_rainfall_rate(dBZ, a=288, b=1.53):
    Z = 10 ** (dBZ/10)
    R = (Z/a) ** (1/b)
    return round(R,2)
    
R_val = calc_rainfall_rate(80) # dbZ value above 60 are related to hail or very large raindrops
print(f"Rainfall rate is {R_val} mm/h")