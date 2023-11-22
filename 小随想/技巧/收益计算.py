class OurEth:
    @staticmethod
    def miner_power(num):
        cp = list(map(float, input("请输入第{}个矿工的各个矿机的算力：".format(num)).split(" ")))
        miner_cp = sum(cp)
        return miner_cp

    def ratio_power(self, day):
        miner_num = int(input("请输入第{}天的矿工数量：".format(day)))
        cp_list = []
        for num in range(miner_num):
            miner_cp = self.miner_power(num + 1)
            cp_list.append(miner_cp)
        ratio_list = []
        for cp in cp_list:
            ratio_list.append(cp / sum(cp_list))
        return miner_num, ratio_list

    def eth_income(self):
        price = int(input("请输入以太币价格："))
        eth_list = list(map(float, input("请输入每天的挖取的以太币数量：").split(" ")))
        wh_eth = []
        zhb_eth = []
        yzh_eth = []
        zl_eth = []
        other_eth = []
        for day, eth in enumerate(eth_list):
            miner_num, ratio_list = self.ratio_power(day + 1)
            if miner_num > 0:
                wh_eth.append(eth * ratio_list[0])
            if miner_num > 1:
                zhb_eth.append(eth * ratio_list[1])
            if miner_num > 2:
                yzh_eth.append(eth * ratio_list[2])
            if miner_num > 3:
                zl_eth.append(eth * ratio_list[3])
            if miner_num > 4:
                other_eth.append(eth * ratio_list[4])
        wh_eth = sum(wh_eth)
        zhb_eth = sum(zhb_eth)
        yzh_eth = sum(yzh_eth)
        zl_eth = sum(zl_eth)
        other_eth = sum(other_eth)
        wh_ic = wh_eth * price
        zhb_ic = zhb_eth * price
        yzh_ic = yzh_eth * price
        zl_ic = zl_eth * price
        other_ic = other_eth * price
        return print(
            "wh_eth:{} wh_ic:{}\nzhb_eth:{} zhb_ic:{}\nyzh_eth:{} yzh_ic:{}\nzl_eth:{} zl_ic:{}\nother_eth:{} other_ic:{}\n".format(
                wh_eth,
                wh_ic,
                zhb_eth,
                zhb_ic,
                yzh_eth,
                yzh_ic,
                zl_eth,
                zl_ic,
                other_eth,
                other_ic,
            )
        )


if __name__ == "__main__":
    eth0606 = OurEth()
    eth0606.eth_income()
