
def valid_dv(delta_vs, dv_max, dv_exception, E):
    exceptions_used = 0

    for dv in delta_vs:
        
        if dv > dv_exception:
            return False
        if dv > dv_max and dv <= dv_exception:
            exceptions_used+=1
        if exceptions_used > E:
            return False

    return True

delta_vs = [80, 150, 120, 650]

print(valid_dv(delta_vs, 100, 600, 5))