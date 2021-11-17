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
            cpu = 10
        elif self.flavor == self.medium:
            cpu = random.randint(6, 9)
        else:
            cpu = random.randint(1, 5)
        return cpu

    def gen_mem(self):
        if self.flavor == self.large:
            mem = 20 if random.randint(1, 2) == 1 else 10
        elif self.flavor == self.medium:
            mem = random.randint(6, 9)
        else:
            mem = random.randint(1, 5)
        return mem


if __name__ == '__main__':
    vnf_generator = VNFGenerator(30, 'large')
    a = vnf_generator.gen_vnf()
    print(a)
