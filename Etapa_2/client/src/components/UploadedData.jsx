import {
	Button,
	Card,
	CardHeader,
	Chip,
	Table,
	TableBody,
	TableCell,
	TableColumn,
	TableHeader,
	TableRow,
} from '@heroui/react';

export default function UploadedData({
	dataInstances,
	actionTitle,
	action,
	isLoading,
}) {
	const cols = {
		Predecir: [
			'ID',
			'Titulo',
			'Descripción',
			'Fecha',
			'Predicción',
			'Probabilidad',
		],
		Reentrenar: ['ID', 'Titulo', 'Descripción', 'Fecha', 'Etiqueta'],
	};

	const rows_retrain = (index, item) => (
		<TableRow key={index} className='items-top'>
			<TableCell className='font-semibold'>{item.ID}</TableCell>
			<TableCell className='font-semibold'>{item.Titulo}</TableCell>
			<TableCell>{item['Descripcion']}</TableCell>
			<TableCell>{String(item.Fecha)}</TableCell>
			<TableCell>
				{item.Label === 1 ? (
					<Chip variant='dot' color='success'>
						Real
					</Chip>
				) : (
					<Chip variant='dot' color='danger'>
						Falsa
					</Chip>
				)}
			</TableCell>
		</TableRow>
	);

	const rows_predict = (index, item) => (
		<TableRow key={index} className='items-top'>
			<TableCell className='font-semibold'>{item.ID}</TableCell>
			<TableCell className='font-semibold'>{item.Titulo}</TableCell>
			<TableCell>{item['Descripcion']}</TableCell>
			<TableCell>{String(item.Fecha)}</TableCell>
			<TableCell>
				{item.prediction === undefined ? (
					' '
				) : item.prediction === 1 ? (
					<Chip variant='dot' color='success'>
						Real
					</Chip>
				) : (
					<Chip variant='dot' color='danger'>
						Falsa
					</Chip>
				)}
			</TableCell>
			<TableCell>
				{item.accuracy === undefined
					? ' '
					: (item.accuracy * 100).toFixed(2) ?? ' '}
				%
			</TableCell>
		</TableRow>
	);

	return (
		<Card className='w-full p-4'>
			<CardHeader className='flex justify-between items-center'>
				<h2 className='text-xl font-semibold'>Datos guardados</h2>
				<Button
					isDisabled={dataInstances.length === 0}
					color='primary'
					variant='shadow'
					onPress={action}
					isLoading={isLoading}
				>
					{actionTitle}
				</Button>
			</CardHeader>
			<Table removeWrapper className='mt-4'>
				<TableHeader>
					{cols[actionTitle].map((column, index) => (
						<TableColumn key={index}>{column}</TableColumn>
					))}
				</TableHeader>
				<TableBody emptyContent={'Agrega datos.'}>
					{dataInstances.map((item, index) =>
						actionTitle === 'Predecir'
							? rows_predict(index, item)
							: rows_retrain(index, item)
					)}
				</TableBody>
			</Table>
		</Card>
	);
}
