import random


class VNFGenerator(object):
    Flavor = (large, medium, small) = ('large', 'medium', 'small')

    def __init__(self, number_of_vnf, flavor):
        self.number_of_vnf = number_of_vnf
        self.flavor = flavor

    def gen_vnf(self):
        vnf_resource_list = list()
        for i in range(self.number_of_vnf):
            vnf_resource = [self.gen_cpu(), self.gen_mem(), 10]
            vnf_resource_list.append(vnf_resource)
        return vnf_resource_list

    def gen_cpu(self):
        if self.flavor == self.large:
            cpu = round(random.randint(5, 6) / 10, 1)
        elif self.flavor == self.medium:
            cpu = round(random.randint(3, 4) / 10, 1)
        else:
            cpu = round(random.randint(1, 2) / 10, 1)
        return cpu

    def gen_mem(self):
        if self.flavor == self.large:
            cpu = round(random.randint(8, 9) / 10, 1) if random.randint(1, 2) == 1 else 1
        elif self.flavor == self.medium:
            cpu = round(random.randint(5, 7) / 10, 1)
        else:
            cpu = round(random.randint(1, 4) / 10, 1)
        return cpu


if __name__ == '__main__':
    vnf_generator = VNFGenerator(50, 'large')
    a = vnf_generator.gen_vnf()
    print(a)
