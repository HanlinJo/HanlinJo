def _handle_transmissions_with_enhanced_sinr(self, transmissions):
    """处理传输并使用SINR进行碰撞检测"""
    # 如果还没有初始化消息状态跟踪器，则创建一个
    if not hasattr(self, 'receiver_message_status'):
        self.receiver_message_status = defaultdict(lambda: {
            'success': False,
            'resources_used': 0,
            'failed_resources': set()
        })
    
    # 按时隙分组
    tx_by_slot = defaultdict(list)
    for sender, packet, resource in transmissions:
        tx_by_slot[resource.slot_id].append((sender, packet, resource))

    collision_info = {'collisions_caused': 0}
    
    # 处理每个时隙的传输
    for slot_id, slot_transmissions in tx_by_slot.items():
        # 检测碰撞：记录每个子信道的使用情况
        subchannel_usage = defaultdict(list)
        for sender, packet, resource in slot_transmissions:
            subchannel_usage[resource.subchannel].append((sender, packet, resource))

        slot_sinr_records = []
        
        # 处理每个子信道
        for subchannel, users in subchannel_usage.items():
            # 检查是否有攻击者参与
            attackers_in_subchannel = [user for user in users if isinstance(user[0], (RLAttacker, FixAttacker))]
            normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
            has_attacker = len(attackers_in_subchannel) > 0

            # 传统碰撞检测：多个发送者使用同一资源块
            collision_occurred = len(normal_users) > 1 or (has_attacker and len(normal_users) > 0)

            # SINR-based接收检测
            if self.use_sinr:
                # 记录该资源块的信息
                resource_record = {
                    'time': self.current_time,
                    'resource': (slot_id, subchannel),
                    'senders': [],
                    'receivers': []
                }

                # 记录发送者信息
                for sender, packet, resource in users:
                    resource_record['senders'].append({
                        'sender_id': sender.id,
                        'sender_position': sender.position.tolist(),
                        'is_attacker': isinstance(sender, (RLAttacker, FixAttacker))
                    })
                
                # 对每个接收者计算SINR
                for receiver in self.vehicles:
                    # 只考虑不是发送者的接收者
                    receiver_sinr_info = {
                        'receiver_id': receiver.id,
                        'receiver_position': receiver.position.tolist(),
                        'sinr_values': [],
                        'sender_ids': [],
                        'distances': []
                    }
                    
                    if not any(sender.id == receiver.id for sender, _, _ in users):
                        for sender, packet, resource in users:
                            if receiver.should_receive_packet(sender.position):
                                # 收集干扰源
                                interferers_pos = []
                                for other_sender, _, _ in users:
                                    if other_sender.id != sender.id:
                                        interferers_pos.append(other_sender.position)
                                
                                # 计算SINR
                                if self.use_simple_interference:
                                    sinr = self.sinr_calculator.calculate_sinr_simple(
                                        receiver.position, sender.position, len(interferers_pos)
                                    )
                                else:
                                    sinr = self.sinr_calculator.calculate_sinr_optimized(
                                        receiver.position, sender.position, interferers_pos
                                    )
                                
                                distance = self._distance(receiver.position, sender.position)
                                
                                # 如果SINR高于阈值，标记为成功接收
                                success = sinr >= self.sinr_threshold
                                
                                # 关键修复：正确处理攻击者造成的干扰
                                if has_attacker and not isinstance(sender, (RLAttacker, FixAttacker)):
                                    # 如果有攻击者在同一子信道，且当前发送者是正常车辆
                                    # 检查是否因为攻击者干扰导致接收失败
                                    if not success:
                                        # 攻击成功：攻击者成功干扰了正常车辆的通信
                                        for attacker_sender, _, _ in attackers_in_subchannel:
                                            if isinstance(attacker_sender, RLAttacker):
                                                attacker_sender.record_attack_success(True)
                                            elif isinstance(attacker_sender, FixAttacker):
                                                attacker_sender.attack_success_count += 1
                                        
                                        collision_info['collisions_caused'] += 1
                                        self.attack_transmission_count += 1
                                
                                if success:
                                    msg_key = (sender.id, packet.packet_id, receiver.id)
                                    self.receiver_message_status[msg_key]['success'] = True
                                else:
                                    # 记录失败原因
                                    msg_key = (sender.id, packet.packet_id, receiver.id)
                                    self.receiver_message_status[msg_key]['failed_resources'].add(receiver.id)

                                # 让接收者处理数据包
                                if isinstance(receiver, Vehicle):
                                    receiver.receive_packet(packet, resource, success)

                                receiver_sinr_info['sinr_values'].append(sinr)
                                receiver_sinr_info['sender_ids'].append(sender.id)
                                receiver_sinr_info['distances'].append(distance)
                    
                    # 只有当接收者有SINR记录时才添加
                    if receiver_sinr_info['sinr_values']:
                        resource_record['receivers'].append(receiver_sinr_info)
                
                # 保存资源块记录
                if resource_record['receivers']:
                    slot_sinr_records.append(resource_record)

                # 让攻击者收集感知数据
                for receiver in self.attackers:
                    if not any(sender.id == receiver.id for sender, _, _ in users):
                        for sender, packet, resource in normal_users:
                            if receiver.should_receive_packet(sender.position):
                                if isinstance(sender, (RLAttacker, FixAttacker)):
                                    pRsvp = 20
                                else:
                                    pRsvp = 100

                                receiver.add_sensing_data(
                                    resource.slot_id,
                                    resource.subchannel,
                                    pRsvp,
                                    sender.id,
                                    packet.timestamp
                                )
            
            else:
                # 不使用SINR时的传统碰撞检测
                for sender, packet, resource in normal_users:
                    # 为每个预期的接收者初始化状态
                    expected_receiver_ids = []
                    for vehicle in self.vehicles:
                        if vehicle.id != sender.id and vehicle.should_receive_packet(sender.position):
                            expected_receiver_ids.append(vehicle.id)

                    for receiver_id in expected_receiver_ids:
                        msg_key = (sender.id, packet.packet_id, receiver_id)
                        if msg_key not in self.receiver_message_status:
                            self.receiver_message_status[msg_key] = {
                                'success': False,
                                'resources_used': 0,
                                'failed_resources': set()
                            }
                        
                        # 更新使用的资源计数
                        self.receiver_message_status[msg_key]['resources_used'] += 1

                        # 传统碰撞检测
                        if has_attacker or collision_occurred:
                            self.receiver_message_status[msg_key]['failed_resources'].add(receiver_id)
                            
                            # 如果是攻击者造成的碰撞，记录攻击成功
                            if has_attacker:
                                for attacker_sender, _, _ in attackers_in_subchannel:
                                    if isinstance(attacker_sender, RLAttacker):
                                        attacker_sender.record_attack_success(True)
                                    elif isinstance(attacker_sender, FixAttacker):
                                        attacker_sender.attack_success_count += 1
                                
                                collision_info['collisions_caused'] += 1
                                self.attack_transmission_count += 1
                        else:
                            # 成功接收
                            self.receiver_message_status[msg_key]['success'] = True

            # 更新资源块级失效原因统计
            if not self.use_sinr:
                if has_attacker:
                    self.resource_block_attacks += 1
                elif collision_occurred:
                    self.resource_block_collisions += 1
        
        self.sinr_records.extend(slot_sinr_records)




    
    # 在所有时隙处理完后，检查完成的消息
    finished_messages = set()
    for msg_key, status in self.receiver_message_status.items():
        sender_id, packet_id, receiver_id = msg_key
        if status['resources_used'] == 2:  # 两个资源块都传输完毕
            # 如果任何资源块失败，整个消息失败
            if status['failed_resources']:
                status['success'] = False
                
            # 更新接收者状态
            receiver = next((v for v in self.vehicles if v.id == receiver_id), None)
            if receiver:
                receiver.record_reception(status['success'])

            # 更新发送者状态
            sender = next((v for v in self.vehicles if v.id == sender_id), None)
            # 更新全局统计
            self.total_expected_packets += 1
            if status['success']:
                self.total_received_packets += 1
            else:
                self.message_failures += 1
                self.collision_count += 1
            if sender and not isinstance(sender, (RLAttacker, FixAttacker)):
            finished_messages.add(msg_key)
                if not status['success']:
    # 移除已完成的消息
    for msg_key in finished_messages:
        del self.receiver_message_status[msg_key]
                    sender.collisions += 1
    return collision_info